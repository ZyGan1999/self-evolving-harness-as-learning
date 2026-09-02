"""Exp 1 (Q1, approximation error): arm runner.

Arms (each = one SessionDriver run; static arms need no training stream):
  baseline   memory none, eval only                       -> no-personalization floor
  oracle1/4/16  memory = persona.oracle_context(v), eval only
             -> isolates approximation error: any residual gap under a perfect,
                arbitrarily long description of f* cannot be estimation/optimization
  learned    assertion updater over an interaction stream, eval at checkpoints
  verifier   control class (call topology): episode-level best-of-n with the
             programmatic preference linter as verifier, eval only
  external   control class (external component): live statistics injected as
             <stats>; runs over a stream so state accumulates, eval at checkpoints
  oracle_verifier  oracle context + verifier gate (upper bound of the two combined)

Usage:
  python scripts/run_exp1.py --llm anthropic:claude-haiku-4-5-20251001 \
      --persona p2_mixed --arms baseline oracle1 learned verifier \
      --n-train 8 --checkpoints 0 4 8 --n-eval 6 [--seed 0] [--tag _pilot]
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from appworld_p.config import OUTPUTS_DIR, PERSONA_DIR, TASK_POOL_PATH  # noqa: E402
from appworld_p.driver import SessionConfig, SessionDriver  # noqa: E402
from appworld_p.persona import Persona  # noqa: E402
from appworld_p.rules import RULE_REGISTRY  # noqa: E402
from appworld_p.streams import build_stream  # noqa: E402
from appworld_p.updaters import AssertionTopL, NoOpUpdater, build_candidate_bank  # noqa: E402

STATIC_ARMS = {"baseline", "oracle1", "oracle4", "oracle16", "verifier",
               "verifier_value", "oracle_verifier",
               "oracle_verifier_value", "oracle_autofill",
               "oracle_stats"}
STREAM_ARMS = {"learned", "external"}


def arm_config(arm: str, run_name: str, persona_path: str, pool: dict,
               args: argparse.Namespace) -> tuple[SessionConfig, object]:
    eval_ids = pool["eval_task_ids"][: args.n_eval]
    base = dict(run_name=run_name, persona_path=persona_path,
                agent=args.agent, llm=args.llm, seed=args.seed,
                feedback_tier=args.feedback_tier, max_steps=args.max_steps,
                eval_task_ids=eval_ids, notes=f"exp1 arm={arm}")
    if arm == "baseline":
        return SessionConfig(stream_task_ids=[], checkpoints=[0],
                             memory_mode="none", **base), NoOpUpdater()
    # oracleN means verbosity N; anything with a suffix is a named arm handled below. An
    # `arm != "oracle_verifier"` guard here silently breaks the moment a second such arm is
    # added -- oracle_verifier_value crashed seed 0 five hours in, after all its episodes
    # had already run, because the index json is only written at the very end.
    if arm.startswith("oracle") and arm.removeprefix("oracle").isdigit():
        v = int(arm.removeprefix("oracle"))
        return SessionConfig(stream_task_ids=[], checkpoints=[0],
                             memory_mode="oracle", oracle_verbosity=v, **base), NoOpUpdater()
    if arm == "verifier":
        return SessionConfig(stream_task_ids=[], checkpoints=[0],
                             memory_mode="none",
                             verifier_attempts=args.verifier_attempts, **base), NoOpUpdater()
    if arm == "verifier_value":
        # same gate, but the checker hands back the value it computed. Isolates detection
        # from computation: reject-only exhausts its attempts on sms_char_checksum because
        # every retry recomputes the same wrong character count.
        return SessionConfig(stream_task_ids=[], checkpoints=[0],
                             memory_mode="none", verifier_detail=True,
                             verifier_attempts=args.verifier_attempts, **base), NoOpUpdater()
    # The information-matched gate arms. These, not the memory_mode="none" ones, carry the
    # control-class claim: a bare `verifier` arm differs from `oracle` in TWO ways (no rule
    # statement AND a gate), so its losing to oracle says nothing about the gate. In p4v2
    # the bare verifier arm ended two episodes sending the literal text 'Done.' -- it never
    # knew a checksum was wanted.
    if arm == "oracle_verifier":
        return SessionConfig(stream_task_ids=[], checkpoints=[0],
                             memory_mode="oracle", oracle_verbosity=1,
                             verifier_attempts=args.verifier_attempts, **base), NoOpUpdater()
    if arm == "oracle_verifier_value":
        return SessionConfig(stream_task_ids=[], checkpoints=[0],
                             memory_mode="oracle", oracle_verbosity=1,
                             verifier_detail=True,
                             verifier_attempts=args.verifier_attempts, **base), NoOpUpdater()
    if arm == "oracle_stats":
        # "harness computes the statistic" for the running-total preference: the ledger is
        # maintained outside f and injected, but f still performs the final addition and writes
        # the field. This is the mechanism that works for the habitual card; running it here
        # tests whether the same strength of harness transfers to an error-intolerant aggregate.
        return SessionConfig(stream_task_ids=[], checkpoints=[0],
                             memory_mode="oracle", oracle_verbosity=1,
                             inject_spend_totals=True, **base), NoOpUpdater()
    if arm == "oracle_autofill":
        # Strongest control-class harness: same oracle statement, but the running total is
        # maintained by the harness and written into the memo at call time. The gate arms show
        # that handing f the expected sequence is not enough on this preference -- it wrote $846
        # into both payments across all three attempts -- so this arm tests whether the residual
        # is arithmetic (which the gate already supplied) or the action space itself.
        return SessionConfig(stream_task_ids=[], checkpoints=[0],
                             memory_mode="oracle", oracle_verbosity=1,
                             autofill_spend_total=True, **base), NoOpUpdater()
    if arm == "learned":
        stream = build_stream(pool, args.n_train, args.seed)
        persona = Persona.load(persona_path)
        persona_rules = [r.name for r in persona.rules]
        distractors = [n for n in RULE_REGISTRY if n not in persona_rules][: args.distractors]
        # shuffle_seed is required, not optional: unshuffled, the bank lists the true rules
        # first and the n=0 memory already contained both of them
        bank = build_candidate_bank(persona_rules, distractors, shuffle_seed=args.seed)
        updater = AssertionTopL(bank, L=args.top_l)
        return SessionConfig(stream_task_ids=stream, checkpoints=args.checkpoints,
                             memory_mode="updater", **base), updater
    if arm == "external":
        stream = build_stream(pool, args.n_train, args.seed)
        return SessionConfig(stream_task_ids=stream, checkpoints=args.checkpoints,
                             memory_mode="none", inject_stats=True, **base), NoOpUpdater()
    raise ValueError(f"unknown arm: {arm}")


def _arm_is_complete(run_name: str, arm: str, args: argparse.Namespace) -> bool:
    """Does this arm already have every eval episode it is supposed to have?

    Counts eval episodes per checkpoint rather than lines in the file: a run killed by a
    relay outage leaves a partial episodes.jsonl, and treating that as done would silently
    report a half-finished arm as a result.
    """
    path = OUTPUTS_DIR / "sessions" / run_name / "episodes.jsonl"
    if not path.exists():
        return False
    try:
        rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    except json.JSONDecodeError:      # truncated final line = interrupted mid-write
        return False
    want_points = args.checkpoints if arm in STREAM_ARMS else [0]
    per_point = Counter(r["n"] for r in rows if r["phase"] == "eval")
    return all(per_point.get(n, 0) >= args.n_eval for n in want_points)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm", required=True)
    parser.add_argument("--persona", required=True)
    parser.add_argument("--agent", default="react", choices=["react", "fc", "oracle"],
                        help="fc = constrained function-call actuator (what Q1 calibrated "
                             "on); oracle = official-solution execution (R044 sensitivity)")
    parser.add_argument("--arms", nargs="+",
                        default=["baseline", "oracle1", "learned",
                                 "oracle_verifier", "oracle_verifier_value"])
    parser.add_argument("--n-train", type=int, default=8)
    parser.add_argument("--checkpoints", nargs="+", type=int, default=[0, 4, 8])
    parser.add_argument("--n-eval", type=int, default=8)
    parser.add_argument("--feedback-tier", default="corrective")
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--verifier-attempts", type=int, default=3)
    # L must be small enough that selection is a real decision. The persona has 2 rules in
    # a bank of 8; L=6 only required excluding 2, so nearly any state kept both true rules.
    parser.add_argument("--top-l", type=int, default=3)
    parser.add_argument("--distractors", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-done", action="store_true",
                        help="skip arms whose eval episodes are already all on disk, so a "
                             "sweep killed by a relay outage can be resumed cheaply")
    parser.add_argument("--tag", default="")
    args = parser.parse_args()

    pool = json.loads(TASK_POOL_PATH.read_text())["pools"][args.persona]
    persona_path = str(PERSONA_DIR / f"{args.persona}.yaml")
    model_slug = args.llm.split(":")[-1].replace(".", "_")

    # the seed MUST be in the run name: without it, replicate runs overwrite each other's
    # session dirs and index json, so seeds 0/1/2 would leave only the last one on disk
    stem = f"exp1{args.tag}_{model_slug}_{args.persona}_s{args.seed}"
    out = OUTPUTS_DIR / f"{stem}.json"
    # Merge into whatever is already on disk instead of starting from {}. A relay outage can
    # kill a sweep 75 minutes in, and the natural recovery is to rerun the missing arms --
    # which used to clobber the index entries of the arms that had already succeeded.
    index = {"llm": args.llm, "persona": args.persona, "seed": args.seed,
             "arms": {}, "args": vars(args)}
    if out.exists():
        try:
            index["arms"] = json.loads(out.read_text()).get("arms", {})
        except (OSError, json.JSONDecodeError) as exc:
            print(f"  !! could not read existing index ({exc}); starting fresh")
    for arm in args.arms:
        run_name = f"{stem}_{arm}"
        if args.skip_done and _arm_is_complete(run_name, arm, args):
            print(f"\n##### arm {arm} -> already complete, skipping")
            continue
        print(f"\n##### arm {arm} -> {run_name}")
        config, updater = arm_config(arm, run_name, persona_path, pool, args)
        manifest = SessionDriver(config, updater).run()
        index["arms"][arm] = {"run_name": run_name, "eval_points": manifest["eval_points"],
                              "llm_usage": manifest["llm_usage"]}
        # flush after every arm: a crash on a later arm used to discard the index for all
        # the arms that had already finished, even though their episodes were on disk
        out.write_text(json.dumps(index, indent=1))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
