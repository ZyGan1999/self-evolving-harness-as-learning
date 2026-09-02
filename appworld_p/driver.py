"""Session driver: runs a persona's interaction stream and checkpoint evaluations.

Loop per training episode:
  load world -> agent (with current memory injected) -> checkers -> weak feedback
  -> updater.observe -> history.update
At each checkpoint n: freeze memory, run frozen eval task set (no feedback, no update).

Everything (config, per-episode rows, updater state trajectory, memory snapshots)
is archived under outputs/sessions/{run_name}/.
"""

import dataclasses
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .autofill import SpendTotalAutofill
from .agent import FunctionCallAgent, OracleAgent, ReactCodeAgent
from .apicalls import resolve_call, was_rejected
from .config import EXPERIMENTS_DIR, OUTPUTS_DIR, set_appworld_root
from .episode import EpisodeRecord
from .feedback import make_feedback
from .habit import HabitStream, pi_u_for
from .history import SessionHistory
from .llm import build_llm
from .tgc import adjudicate
from .persona import Persona
from .rules import RULE_REGISTRY
from .updaters import BaseUpdater


REJECT_HEADER = ("Automated constraint checker REJECTED the previous attempt at this "
                 "task for these reasons — you must comply this time:\n")

CARD_LOOKUP_CODE = """
_profile = apis.supervisor.show_profile()
_pw = {p['account_name']: p['password'] for p in apis.supervisor.show_account_passwords()}
_tok = apis.venmo.login(username=_profile['email'], password=_pw['venmo'])['access_token']
print([(c['payment_card_id'], c['card_name']) for c in apis.venmo.show_payment_cards(access_token=_tok)])
"""


SPEND_LOOKUP_CODE = """
_profile = apis.supervisor.show_profile()
_me = _profile['email']
_pw = {p['account_name']: p['password'] for p in apis.supervisor.show_account_passwords()}
_tok = apis.venmo.login(username=_profile['email'], password=_pw['venmo'])['access_token']
_all, _pg = [], 0
while True:
    _batch = apis.venmo.show_transactions(access_token=_tok, page_index=_pg, page_limit=20)
    if not _batch or _pg > 40:
        break
    _all += _batch
    _pg += 1
_sent = [t for t in _all if t['sender']['email'] == _me]
_per = {}
for _t in _sent:
    _per[_t['receiver']['email']] = round(_per.get(_t['receiver']['email'], 0.0)
                                         + float(_t['amount']), 2)
print([round(sum(float(t['amount']) for t in _sent), 2), len(_sent), _per])
"""


def render_spend_stats(spend_totals: dict) -> str:
    """The harness-computed statistic for the running-total preference.

    This is the `usual_card` counter's analogue: the aggregate f cannot hold reliably is computed
    outside f and handed over, while f still decides what to do with it. The contrast the pair is
    built for is in the functional form -- the card counter feeds an argmax (error-tolerant), this
    feeds an exact sum (error-intolerant), so if injection rescues one and not the other, the
    difference is the shape of the target, not the amount of information.

    It states the pre-episode figure per recipient and says explicitly that this episode's
    payments accumulate on top -- but it does not do that addition, which is what separates this
    arm from autofill. Everything here is already reachable through the paginated API the agent
    can call itself; the harness only saves it the traversal.
    """
    per = spend_totals.get("per_recipient") or {}
    if not per:
        return ""
    lines = ["Ledger maintained for you (Venmo money you have already sent, before this task):"]
    for email, amount in sorted(per.items(), key=lambda kv: -kv[1]):
        lines.append(f"  {email}: ${amount:g} sent so far")
    lines.append(f"Across all recipients: ${spend_totals.get('total', 0):g} in "
                 f"{spend_totals.get('count', 0)} payments.")
    lines.append("These figures EXCLUDE anything you send during this task. If you pay someone "
                 "twice here, the second payment's cumulative figure must include the first.")
    return "\n".join(lines)


def read_spend_totals(world) -> dict:
    """Venmo money the supervisor has already sent, before the agent acts.

    -> {"total": float, "count": int, "per_recipient": {email: float}}. Read-only and
    checker-side: the agent never sees this, it has to derive the same figure through the
    same paginated API the lookup uses.
    """
    import ast
    try:
        raw = world.execute(SPEND_LOOKUP_CODE).strip()
    except Exception as exc:  # a lookup failure must not kill the episode
        print(f"    !! spend lookup failed: {exc}", flush=True)
        return {}
    for parse in (ast.literal_eval, json.loads):
        try:
            total, count, per = parse(raw)
            return {"total": float(total), "count": int(count),
                    "per_recipient": {str(k): float(v) for k, v in per.items()}}
        except Exception:
            continue
    print(f"    !! spend lookup unparseable: {raw[:120]!r}", flush=True)
    return {}


def read_card_names(world) -> dict[str, str]:
    """{payment_card_id: bank name} for this world. Read-only, runs before the agent
    (a checker-side lookup, not part of the agent's observation)."""
    import ast
    try:
        raw = world.execute(CARD_LOOKUP_CODE).strip()
    except Exception as exc:  # a lookup failure must not kill the episode
        print(f"    !! card lookup failed: {exc}", flush=True)
        return {}
    for parse in (ast.literal_eval, json.loads):
        try:
            return {str(cid): name for cid, name in parse(raw)}
        except Exception:
            continue
    print(f"    !! card lookup unparseable: {raw[:120]!r}", flush=True)
    return {}


@dataclass
class SessionConfig:
    run_name: str
    persona_path: str
    stream_task_ids: list[str]
    eval_task_ids: list[str] = field(default_factory=list)
    checkpoints: list[int] = field(default_factory=list)   # evaluate after these many train episodes
    agent: str = "oracle"            # "oracle" | "react" | "fc" (constrained actuator)
    llm: str = "mock"                # llm spec for react agent (see llm.build_llm)
    feedback_tier: str = "corrective"
    max_steps: int = 30
    seed: int = 0
    notes: str = ""
    extra_rules: tuple[str, ...] = ()   # scored on top of the persona's own rules:
                                        # calibration measures a RULE, which need not
                                        # belong to any persona
    # --- harness arm configuration (Exp-1) ---
    memory_mode: str = "updater"     # "updater" | "none" | "oracle" | "fixed" | "fulllog"
    oracle_verbosity: int = 1        # for memory_mode="oracle": persona.oracle_context(v)
    fixed_memory: str = ""           # for memory_mode="fixed"
    eval_label_n: int | None = None  # label eval rows with THIS n instead of the count of train
                                     # episodes actually run. Only for backfilling a checkpoint
                                     # into a finished evolving run: the memory for stream
                                     # position n is replayed from that run's own audit trail and
                                     # injected via memory_mode="fixed", so no training is
                                     # re-executed and the new point lands on the SAME memory
                                     # trajectory as the original checkpoints. Re-running the
                                     # stream instead would resample at temperature 0.7 and put
                                     # the new point on a different trajectory than its
                                     # neighbours. See scripts/replay_ace_memory.py.
    inject_stats: bool = False       # control class: external counter injected as <stats>
    # --- family B: statistical-aggregation habit stream (see appworld_p/habit.py) ---
    habit_dominant: str = ""         # bank name carrying most of pi_u ("" = habit off)
    habit_tier: str = "binary"       # "binary" (out regime) | "corrective" (in regime)
    habit_noisy: bool = False        # interleave unrelated interaction lines
    habit_noise_per_round: float = 1.0   # dilution density: mean noise lines per round
    habit_schedule: list[int] = field(default_factory=list)
                                     # cumulative habit-observation counts to evaluate at,
                                     # e.g. [0, 20, 60, 120] — the log-length sweep
    inject_spend_totals: bool = False  # control class: the harness computes the per-recipient
                                     # running total and injects it as a statistic; f still
                                     # decides what to write. The counter analogue of
                                     # inject_stats, for an error-INTOLERANT aggregate.
    autofill_spend_total: bool = False  # control class: the harness maintains the running
                                     # per-recipient total and rewrites the memo tag at call
                                     # time, so f never has to hold the sum. FC actuator only.
    verifier_attempts: int = 1       # control class: episode-level best-of-n with checker
                                     # verifier (1 = gate off)
    self_gate_attempts: int = 1      # TRACE (arXiv:2606.13174): retry against the UPDATER's own
                                     # compiled checks, not the persona's. Separate from
                                     # verifier_attempts on purpose -- this gate is part of the
                                     # recipe under test, and enforcing with the persona checkers
                                     # would hand that arm the ground truth every other arm is
                                     # denied, silently making it a control-class arm. Requires
                                     # the updater to expose check(ep) -> list[str].
    verifier_nonleaking: bool = False  # reject-only gate that uses Rule.gate_detail instead of
                                     # `detail`: says what the attempt did, never the target.
                                     # For rules whose `detail` names the answer, this is the
                                     # only admissible way to run a reject-only arm.
    verifier_accumulate: bool = False  # carry every attempt's rejection forward instead of
                                     # replacing it. Retries get a fresh context, so without
                                     # this the agent forgets which options were already
                                     # excluded and the gate measures memory, not mechanism.
    verifier_feedback: bool = True   # pass the linter's violation report into retry
                                     # attempts (checker output flowing through the call
                                     # topology — knowledge via program state, not learning)
    verifier_detail: bool = False     # ALSO hand back the checker's computed expectation
                                     # ('expected 1 for 393 chars'), not just a restatement
                                     # of the rule. Separates detecting a violation from
                                     # supplying the computation the model cannot do: on
                                     # sms_char_checksum the reject-only gate exhausts all
                                     # 3 attempts and still fails, because each retry
                                     # recomputes the same wrong character count.


class SessionDriver:
    def __init__(self, config: SessionConfig, updater: BaseUpdater):
        set_appworld_root()
        self.config = config
        self.updater = updater
        self.persona = Persona.load(config.persona_path)
        for name in config.extra_rules:
            if name not in {r.name for r in self.persona.rules}:
                self.persona.rules.append(RULE_REGISTRY[name]())
        self.history = SessionHistory()
        self.habit: HabitStream | None = None
        if config.habit_dominant:
            self.habit = HabitStream(pi_u=pi_u_for(config.habit_dominant),
                                     tier=config.habit_tier, seed=config.seed,
                                     noisy=config.habit_noisy,
                                     noise_per_round=config.habit_noise_per_round)
            self.history.habit_target = self.habit.target()
        self.llm = build_llm(config.llm) if config.agent in ("react", "fc") else None
        self.out_dir = OUTPUTS_DIR / "sessions" / config.run_name
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.rows: list[dict] = []
        self._correction_templates = {r.name: r.correction_template for r in self.persona.rules}
        # only pay the extra world.execute() when a rule actually speaks bank names
        self._needs_card_names = any(r.name == "usual_card" for r in self.persona.rules)
        # same idea for the exact-aggregation rules: paginating the whole transaction list is
        # expensive, so only do it when a rule is defined against that total
        self._needs_spend_totals = any(r.name.startswith("spend_total_")
                                       for r in self.persona.rules)
        # progress tracking
        n_checkpoints = len(config.habit_schedule or config.checkpoints)
        self._total_episodes = (len(config.stream_task_ids)
                                + len(config.eval_task_ids) * n_checkpoints)
        self._done_episodes = 0
        self._t0 = time.time()

    def _progress(self, label: str) -> None:
        self._done_episodes += 1
        elapsed = time.time() - self._t0
        rate = elapsed / self._done_episodes
        remaining = rate * (self._total_episodes - self._done_episodes)
        print(f"  [{self._done_episodes}/{self._total_episodes}] {label} | "
              f"elapsed {elapsed/60:.1f}m | eta {remaining/60:.1f}m", flush=True)

    # ------------------------------------------------------------------ episodes
    def _current_memory(self) -> str:
        mode = self.config.memory_mode
        if mode == "updater":
            return self.updater.render_memory()
        if mode == "none":
            return ""
        if mode == "oracle":
            return self.persona.oracle_context(self.config.oracle_verbosity)
        if mode == "fixed":
            return self.config.fixed_memory
        if mode == "fulllog":
            # context-class information upper bound: the adoptable rule statement plus
            # the complete raw interaction log (grows with n; no statistic computed)
            parts = [self.persona.oracle_context(self.config.oracle_verbosity)]
            if self.habit and self.habit.rounds:
                parts.append("Record of my past card payments and how I reacted:\n"
                             + self.habit.render_log())
            return "\n\n".join(parts)
        raise ValueError(f"unknown memory_mode: {mode}")

    def _run_episode_once(self, task_id: str, phase: str, session_index: int,
                          experiment_name: str, extra_memory: str = "") -> EpisodeRecord:
        from appworld import AppWorld  # deferred: needs APPWORLD_ROOT set

        memory = self._current_memory()
        if extra_memory:
            memory = (memory + "\n\n" + extra_memory).strip()
        external_stats = ""
        if self.config.inject_stats:
            # family B: the acceptance-rate counter over the habit log; otherwise the
            # generic session counters (running counts, recent cards)
            external_stats = (self.habit.external_stats() if self.habit
                              else self.history.render_external_stats())
        with AppWorld(task_id=task_id, experiment_name=experiment_name,
                      ground_truth_mode="full", raise_on_failure=False) as world:
            # read-only lookup BEFORE the agent runs: world-local card ids -> bank names
            card_names = read_card_names(world) if self._needs_card_names else {}
            spend_totals = read_spend_totals(world) if self._needs_spend_totals else {}
            if self.config.inject_spend_totals:
                # must be built here, not above: the figure it states is world-local and only
                # available once the episode's world is open
                external_stats = "\n\n".join(
                    x for x in (external_stats, render_spend_stats(spend_totals)) if x)
            if self.config.agent == "oracle":
                agent_info = OracleAgent().solve(world)
            else:
                agent_cls = {"react": ReactCodeAgent, "fc": FunctionCallAgent}[self.config.agent]
                kw = {}
                if self.config.autofill_spend_total:
                    if self.config.agent != "fc":
                        raise ValueError("autofill_spend_total requires the fc actuator")
                    # seeded from the same read-only lookup the rule scores against, so the
                    # harness cannot be right for a reason the checker disagrees with
                    kw["autofill"] = SpendTotalAutofill(
                        baseline=dict(spend_totals.get("per_recipient", {})))
                agent = agent_cls(self.llm, max_steps=self.config.max_steps,
                                  memory=memory, external_stats=external_stats,
                                  verbose=True, **kw)
                agent_info = agent.solve(world)
            completed = bool(world.task_completed())
            try:
                evaluation = world.evaluate().to_dict()
            except Exception as exc:  # evaluation must never kill the stream
                evaluation = {"error": str(exc)}
            api_calls = [c for c in (resolve_call(dict(r))
                                     for r in world.requester.request_tracker.requests)
                         if not was_rejected(c)]
            supervisor = dict(world.task.supervisor)
            instruction = world.task.instruction
        tgc = evaluation.get("success")
        # Official TGC fails whenever a preference decorates a field AppWorld asserts exactly,
        # so it cannot be compared across arms as-is; adjudicate now rather than leaving every
        # run to need a scripts/rescore_tgc.py pass afterwards.
        # tgc is None when evaluation itself failed; keep the relaxed verdict None too rather
        # than reporting a decisive False for an episode that was never scored
        verdict = (adjudicate(evaluation, [r.name for r in self.persona.rules])
                   if tgc is not None else None)
        transcript = agent_info.pop("transcript", None)
        if transcript is not None:  # raw LLM replies -> session dir (debugging relays etc.)
            tdir = self.out_dir / "transcripts"
            tdir.mkdir(exist_ok=True)
            (tdir / f"{experiment_name}.json").write_text(json.dumps(
                {"task_id": task_id, "system_prompt": agent_info.pop("system_prompt", ""),
                 "steps": transcript}, indent=1))
        return EpisodeRecord(task_id=task_id, instruction=instruction, supervisor=supervisor,
                             api_calls=api_calls, task_completed=completed, tgc=tgc,
                             tgc_relaxed=verdict.relaxed if verdict else None,
                             tgc_excused=verdict.excused if verdict else [],
                             session_index=session_index, phase=phase,
                             card_names=card_names, spend_totals=spend_totals,
                             meta={"agent": agent_info, "evaluation": evaluation,
                                   "memory_at_start": memory,
                                   "external_stats": external_stats})

    def _run_episode_self_gated(self, task_id: str, phase: str, session_index: int,
                                experiment_name: str) -> EpisodeRecord:
        """TRACE's gate: retry against the UPDATER's own compiled checks.

        The distinction from the verifier gate is the source of the verdict. Here it is the
        learner's own rules, mined from the complaints it has seen, so the arm gets no information
        the other context-class arms lack -- it only gets to act on what it already believes
        before finishing. Enforcing with `self.persona.check` instead would leak the ground truth
        and turn this into a control-class arm.

        A compiled check on a preference the task makes unsatisfiable would block forever, so the
        loop is bounded and the last attempt is returned regardless. `self_gate_blocked` records
        how many attempts its own rules rejected, which is what separates "the rules were
        satisfied" from "the rules were never satisfiable".
        """
        gate = getattr(self.updater, "check", None)
        attempts = max(1, self.config.self_gate_attempts)
        ep = None
        report = ""
        blocked = 0
        for attempt in range(1, attempts + 1):
            suffix = f"_g{attempt}" if attempts > 1 else ""
            ep = self._run_episode_once(task_id, phase, session_index,
                                        experiment_name + suffix, extra_memory=report)
            ep.meta["self_gate_attempt"] = attempt
            if gate is None:
                break
            failures = gate(ep)
            if not failures:
                break
            blocked += 1
            report = ("Your own recorded rules rejected this attempt before you could finish:\n"
                      + "\n".join(f"- {f}" for f in failures)
                      + "\nRedo the task so these pass.")
        if ep is not None:
            ep.meta["self_gate_blocked"] = blocked
        return ep

    def _run_episode(self, task_id: str, phase: str, session_index: int,
                     experiment_name: str) -> EpisodeRecord:
        """Episode with optional verifier gate: sample up to k attempts (fresh world
        each time), accept the first one that passes the preference linter (a
        control-class harness: deterministic checker in the call topology).

        A clean attempt is not the same as an attempt with nothing to check. The habitual-card
        gate exposed the difference: rejected on attempt 1, the agent spent attempt 2's whole
        step budget re-reading the card list and never sent the payment, so no rule was
        applicable, no rule was violated, and the naive loop accepted it -- the constraint
        satisfied by declining to act, and the episode then dropped out of the denominator
        instead of counting as a miss. So an attempt that skipped the constrained action is not
        treated as compliance: retry, and if nothing better arrives prefer the attempt that
        acted and failed over the one that dodged, which is the truthful record of what the gate
        achieved. `escaped` counts the dodges for audit.
        """
        attempts = max(1, self.config.verifier_attempts)
        if attempts == 1 and self.config.self_gate_attempts > 1:
            return self._run_episode_self_gated(task_id, phase, session_index, experiment_name)
        ep = None
        linter_report = ""
        acted_fallback = None   # best attempt that actually took the constrained action
        escaped = 0
        self._gate_lines: list[str] = []   # accumulated exclusions, oldest first
        for attempt in range(1, attempts + 1):
            suffix = f"_a{attempt}" if attempts > 1 else ""
            ep = self._run_episode_once(task_id, phase, session_index,
                                        experiment_name + suffix,
                                        extra_memory=linter_report)
            if attempts == 1:
                break
            results = self.persona.check(ep, self.history)
            violations = [r for r in results if r.applicable and r.satisfied is False]
            acted = any(r.applicable for r in results)
            ep.meta["verifier_attempt"] = attempt
            if acted and not violations:
                break                      # genuinely compliant: this is what the gate is for
            if acted:
                acted_fallback = ep         # took the action and missed; keep as the honest record
            else:
                escaped += 1
                if not violations:
                    # nothing to reject and nothing done -- retry rather than bank it as a pass
                    continue
            if self.config.verifier_feedback:
                if self.config.verifier_accumulate:
                    # One header over a growing bullet list. Concatenating whole reports instead
                    # repeats "checker REJECTED the previous attempt" once per attempt, which
                    # reads as several separate verdicts on the same try.
                    for line in self._report_lines(violations, ep):
                        if line not in self._gate_lines:
                            self._gate_lines.append(line)
                    linter_report = REJECT_HEADER + "\n".join(
                        f"- {ln}" for ln in self._gate_lines)
                else:
                    linter_report = self._linter_report(violations, ep)
        if ep is not None and attempts > 1:
            if acted_fallback is not None and not any(
                    r.applicable for r in self.persona.check(ep, self.history)):
                ep = acted_fallback     # never let a dodge outrank an attempt that acted
            ep.meta["gate_escapes"] = escaped
        return ep

    def _report_lines(self, violations: list, ep=None) -> list[str]:
        """One line per violation: the correction, plus whichever detail channel is configured."""
        lines = []
        for v in violations:
            line = self._correction_templates.get(v.rule, v.rule)
            if self.config.verifier_nonleaking and ep is not None:
                # non-leaking channel: the rule describes the attempt, not the target
                rule_obj = next((r for r in self.persona.rules if r.name == v.rule), None)
                hint = rule_obj.gate_detail(ep, self.history) if rule_obj else ""
                if hint:
                    line = f"{line} Checker observed: {hint}"
            elif self.config.verifier_detail and v.detail:
                line = f"{line} Checker measured: {v.detail}"
            lines.append(line)
        return lines

    def _linter_report(self, violations: list, ep=None) -> str:
        """Checker output fed into the next attempt. With verifier_detail the checker's
        computed expectation goes back too, which is what separates detecting a violation
        from supplying a computation the model cannot perform."""
        return REJECT_HEADER + "\n".join(f"- {ln}" for ln in self._report_lines(violations, ep))

    def _first_card_bank(self, ep: EpisodeRecord) -> str | None:
        """Bank of the first card the episode tried (family-B choice distribution)."""
        from .rules.api_map import find_actions
        for call in find_actions(ep, "venmo_payment"):
            card = call.arg("payment_card_id", "card_id")
            if card is not None:
                return ep.card_names.get(str(card), f"id={card}")
        return None

    def _record(self, ep: EpisodeRecord, results, n_seen: int) -> dict:
        row = {
            "phase": ep.phase, "n": n_seen, "session_index": ep.session_index,
            "task_id": ep.task_id, "task_completed": ep.task_completed, "tgc": ep.tgc,
            "tgc_relaxed": ep.tgc_relaxed, "tgc_excused": ep.tgc_excused,
            "n_api_calls": len(ep.api_calls),
            "habit_n": self.history.habit_rounds if self.habit else None,
            "habit_target": self.history.habit_target,
            "first_card": self._first_card_bank(ep) if self._needs_card_names else None,
            "verifier_attempt": ep.meta.get("verifier_attempt"),
            # How many fields the autofill harness actually rewrote. Needed to tell "the
            # harness fixed it" apart from "the agent happened to be right": a 0.00 violation
            # rate with zero rewrites means the arm never did anything.
            "autofill_repairs": len(ep.meta.get("agent", {}).get("autofill_repairs", [])),
            # attempts where the gate had nothing to check because the constrained action never
            # happened. A gate arm whose rate looks good on a shrunken denominator is explained
            # by this column, so it travels with the numbers rather than being reconstructed.
            "gate_escapes": ep.meta.get("gate_escapes"),
            "rules": {r.rule: {"applicable": r.applicable, "satisfied": r.satisfied,
                               "detail": r.detail} for r in results},
            "violations": sum(1 for r in results if r.applicable and r.satisfied is False),
            "applicable": sum(1 for r in results if r.applicable),
        }
        self.rows.append(row)
        return row

    # ---------------------------------------------------------------------- run
    def run(self) -> dict:
        cfg = self.config
        started = time.time()
        eval_points: dict[str, dict] = {}
        print(f"Session {cfg.run_name}: {len(cfg.stream_task_ids)} train + "
              f"{len(cfg.eval_task_ids)} eval x {len(cfg.checkpoints)} checkpoints "
              f"= {self._total_episodes} episodes | agent={cfg.agent} llm={cfg.llm}", flush=True)

        def run_eval(n_seen: int) -> None:
            # A backfilled checkpoint runs zero train episodes but must be recorded at the stream
            # position whose memory it was given, or it lands at n=0 and reads as a second
            # measurement of the empty-memory baseline.
            label = cfg.eval_label_n if cfg.eval_label_n is not None else n_seen
            stats = {"episodes": 0, "applicable": 0, "violations": 0, "tgc_pass": 0}
            for j, task_id in enumerate(cfg.eval_task_ids):
                print(f"  eval@n={label} task {task_id} ...", flush=True)
                ep = self._run_episode(task_id, phase="eval", session_index=-1,
                                       experiment_name=f"apw_p_{cfg.run_name}_eval{label}_{j}")
                results = self.persona.check(ep, self.history)
                self._record(ep, results, label)
                stats["episodes"] += 1
                stats["applicable"] += sum(1 for r in results if r.applicable)
                stats["violations"] += sum(1 for r in results if r.applicable and not r.satisfied)
                stats["tgc_pass"] += int(bool(ep.tgc))
                viol = sum(1 for r in results if r.applicable and not r.satisfied)
                self._progress(f"eval@n={label} {task_id} tgc={ep.tgc} violations={viol}")
            stats["violation_rate"] = (stats["violations"] / stats["applicable"]
                                       if stats["applicable"] else None)
            eval_points[str(label)] = stats  # str keys: identical in memory and on disk
            print(f"  == checkpoint n={label}: violation_rate="
                  f"{stats['violation_rate']} tgc={stats['tgc_pass']}/{stats['episodes']} ==",
                  flush=True)
            # dump what was ACTUALLY injected at this checkpoint, not the updater's
            # state: for oracle/fulllog/fixed arms the updater is a no-op, so dumping
            # render_memory() wrote an empty file and left no record of the context
            (self.out_dir / f"memory_n{label}.md").write_text(self._current_memory())
            if self.config.inject_stats:
                (self.out_dir / f"stats_n{label}.md").write_text(
                    self.habit.external_stats() if self.habit
                    else self.history.render_external_stats())

        # --- family B: habit-observation sweep instead of an AppWorld task stream ---
        # The habit stream is synthetic micro-rounds (zero LLM cost, same construction as
        # the skill-selection paper's synthetic user); the x-axis is observations, not
        # episodes. All arms see the identical stream; only the harness differs.
        if self.habit is not None and cfg.habit_schedule:
            print(f"  habit stream: dominant={cfg.habit_dominant} tier={cfg.habit_tier} "
                  f"target={self.habit.target()} schedule={cfg.habit_schedule}", flush=True)
            for n_obs in cfg.habit_schedule:
                self.habit.advance(n_obs)
                self.history.habit_rounds = len(self.habit.rounds)
                print(f"  -- habit n={n_obs}: {self.habit.state()}", flush=True)
                run_eval(n_obs)
            self.habit.save(self.out_dir / "habit.json")
            return self._finish(started, eval_points)

        if 0 in cfg.checkpoints:
            run_eval(0)
        for i, task_id in enumerate(cfg.stream_task_ids, start=1):
            print(f"  train {i}/{len(cfg.stream_task_ids)} task {task_id} ...", flush=True)
            ep = self._run_episode(task_id, phase="train", session_index=i,
                                   experiment_name=f"apw_p_{cfg.run_name}_train{i}")
            results = self.persona.check(ep, self.history)
            # rules_by_name is only read by the vague_scoped tier, which derives its scope hint
            # from each violated rule's trigger_actions rather than from authored per-rule text.
            feedback = make_feedback(results, tier=cfg.feedback_tier,
                                     correction_templates=self._correction_templates,
                                     rules_by_name={r.name: r for r in self.persona.rules})
            self.updater.observe(ep, feedback)
            self.history.update(ep)
            row = self._record(ep, results, i)
            row["feedback_accepted"] = feedback.accepted
            row["updater_state"] = self.updater.state()
            self._progress(f"train {i} {task_id} tgc={ep.tgc} "
                           f"accepted={feedback.accepted} steps={ep.meta['agent'].get('steps')}")
            if i in cfg.checkpoints:
                run_eval(i)

        return self._finish(started, eval_points)

    def _finish(self, started: float, eval_points: dict) -> dict:
        manifest_config = dataclasses.asdict(self.config)
        persona_path = Path(manifest_config["persona_path"])
        try:
            manifest_config["persona_path"] = str(
                persona_path.resolve().relative_to(EXPERIMENTS_DIR.resolve()))
        except ValueError:
            manifest_config["persona_path"] = persona_path.name
        manifest = {
            "config": manifest_config,
            "persona": {"name": self.persona.name,
                        "rules": [r.name for r in self.persona.rules]},
            "eval_points": eval_points,
            "habit": self.habit.state() if self.habit else None,
            "duration_s": round(time.time() - started, 1),
            "llm_usage": dataclasses.asdict(self.llm.usage) if self.llm else None,
        }
        (self.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1))
        (self.out_dir / "episodes.jsonl").write_text(
            "\n".join(json.dumps(r, default=str) for r in self.rows))
        self.history.save(self.out_dir / "history.json")
        return manifest
