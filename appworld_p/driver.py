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
from .apicalls import resolve_call, was_rejected, track_payment_outcomes
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




def read_spend_totals(world) -> dict:
    """Read the initial outgoing-payment totals for checking and autofill.

    Returns total, count, and per-recipient amounts.
    """
    import ast
    try:
        raw = world.execute(SPEND_LOOKUP_CODE).strip()
    except Exception as exc:
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
    except Exception as exc:
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
    checkpoints: list[int] = field(default_factory=list)
    agent: str = "oracle"
    llm: str = "mock"
    feedback_tier: str = "corrective"
    max_steps: int = 30
    seed: int = 0
    notes: str = ""
    extra_rules: tuple[str, ...] = ()

    memory_mode: str = "updater"
    oracle_verbosity: int = 1
    fixed_memory: str = ""
    eval_label_n: int | None = None

    inject_stats: bool = False

    habit_dominant: str = ""
    habit_tier: str = "binary"
    habit_noisy: bool = False
    habit_noise_per_round: float = 1.0
    habit_schedule: list[int] = field(default_factory=list)

    autofill_spend_total: bool = False

    verifier_attempts: int = 1  # Lightweight control using the persona checker.

    self_gate_attempts: int = 1  # TRACE retries use only its learned checks.

    verifier_nonleaking: bool = False

    verifier_accumulate: bool = False

    verifier_feedback: bool = True

    verifier_detail: bool = False  # Return computed values only in value-feedback arms.


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

        self._needs_card_names = any(r.name == "usual_card" for r in self.persona.rules)

        self._needs_spend_totals = any(r.name.startswith("spend_total_")
                                       for r in self.persona.rules)

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
            parts = [self.persona.oracle_context(self.config.oracle_verbosity)]
            if self.habit and self.habit.rounds:
                parts.append("Record of my past card payments and how I reacted:\n"
                             + self.habit.render_log())
            return "\n\n".join(parts)
        raise ValueError(f"unknown memory_mode: {mode}")

    def _run_episode_once(self, task_id: str, phase: str, session_index: int,
                          experiment_name: str, extra_memory: str = "") -> EpisodeRecord:
        from appworld import AppWorld

        memory = self._current_memory()
        if extra_memory:
            memory = (memory + "\n\n" + extra_memory).strip()
        external_stats = ""
        if self.config.inject_stats:
            external_stats = (self.habit.external_stats() if self.habit
                              else self.history.render_external_stats())
        with AppWorld(task_id=task_id, experiment_name=experiment_name,
                      ground_truth_mode="full", raise_on_failure=False) as world:
            card_names = read_card_names(world) if self._needs_card_names else {}
            spend_totals = read_spend_totals(world) if self._needs_spend_totals else {}
            if self.config.agent == "oracle":
                with track_payment_outcomes(world.requester):
                    agent_info = OracleAgent().solve(world)
            else:
                agent_cls = {"react": ReactCodeAgent, "fc": FunctionCallAgent}[self.config.agent]
                kw = {}
                if self.config.autofill_spend_total:
                    if self.config.agent != "fc":
                        raise ValueError("autofill_spend_total requires the fc actuator")

                    kw["autofill"] = SpendTotalAutofill(
                        baseline=dict(spend_totals.get("per_recipient", {})))
                agent = agent_cls(self.llm, max_steps=self.config.max_steps,
                                  memory=memory, external_stats=external_stats,
                                  verbose=True, **kw)
                with track_payment_outcomes(world.requester):
                    agent_info = agent.solve(world)
            completed = bool(world.task_completed())
            try:
                evaluation = world.evaluate().to_dict()
            except Exception as exc:
                evaluation = {"error": str(exc)}
            api_calls = [c for c in (resolve_call(dict(r))
                                     for r in world.requester.request_tracker.requests)
                         if not was_rejected(c)]
            supervisor = dict(world.task.supervisor)
            instruction = world.task.instruction
        tgc = evaluation.get("success")

        verdict = (adjudicate(evaluation, [r.name for r in self.persona.rules])
                   if tgc is not None else None)
        transcript = agent_info.pop("transcript", None)
        if transcript is not None:
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
        """Retry using the updater's learned checks, with a fresh task world per attempt.
        The persona checker is used only for subsequent evaluation, not for this gate."""
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
        """Run up to the configured number of fresh-world attempts using the persona checker.
        Accept the first applicable, compliant attempt. At exhaustion, prefer an acted
        fallback when the last attempt contains no applicable action."""
        attempts = max(1, self.config.verifier_attempts)
        if attempts == 1 and self.config.self_gate_attempts > 1:
            return self._run_episode_self_gated(task_id, phase, session_index, experiment_name)
        ep = None
        linter_report = ""
        acted_fallback = None
        escaped = 0
        self._gate_lines: list[str] = []
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
                break
            if acted:
                acted_fallback = ep
            else:
                escaped += 1
                if not violations:
                    continue
            if self.config.verifier_feedback:
                if self.config.verifier_accumulate:
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
                ep = acted_fallback
            ep.meta["gate_escapes"] = escaped
        return ep

    def _report_lines(self, violations: list, ep=None) -> list[str]:
        """One line per violation: the correction, plus whichever detail channel is configured."""
        lines = []
        for v in violations:
            line = self._correction_templates.get(v.rule, v.rule)
            if self.config.verifier_nonleaking and ep is not None:
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
        computed expectation goes back too."""
        return REJECT_HEADER + "\n".join(f"- {ln}" for ln in self._report_lines(violations, ep))

    def _first_card_bank(self, ep: EpisodeRecord) -> str | None:
        """Bank of the first payment card attempted in the episode."""
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

            "autofill_repairs": len(ep.meta.get("agent", {}).get("autofill_repairs", [])),

            "gate_escapes": ep.meta.get("gate_escapes"),
            "rules": {r.rule: {"applicable": r.applicable, "satisfied": r.satisfied,
                               "detail": r.detail} for r in results},
            "violations": sum(1 for r in results if r.applicable and r.satisfied is False),
            "applicable": sum(1 for r in results if r.applicable),
        }
        self.rows.append(row)
        return row

    def run(self) -> dict:
        cfg = self.config
        started = time.time()
        eval_points: dict[str, dict] = {}
        print(f"Session {cfg.run_name}: {len(cfg.stream_task_ids)} train + "
              f"{len(cfg.eval_task_ids)} eval x {len(cfg.checkpoints)} checkpoints "
              f"= {self._total_episodes} episodes | agent={cfg.agent} llm={cfg.llm}", flush=True)

        def run_eval(n_seen: int) -> None:
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
            eval_points[str(label)] = stats
            print(f"  == checkpoint n={label}: violation_rate="
                  f"{stats['violation_rate']} tgc={stats['tgc_pass']}/{stats['episodes']} ==",
                  flush=True)

            (self.out_dir / f"memory_n{label}.md").write_text(self._current_memory())
            if self.config.inject_stats:
                (self.out_dir / f"stats_n{label}.md").write_text(
                    self.habit.external_stats() if self.habit
                    else self.history.render_external_stats())

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
