"""Run Q1 context and control harnesses.

Context: no memory, stated preferences, or history-derived assertions.
Control: checker-guided retry, external statistics, or action-field computation.
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

STATIC_ARMS = {"baseline", "oracle1", "oracle_verifier",
               "oracle_verifier_value", "oracle_autofill",
               "oracle_stats"}
STREAM_ARMS = {"learned"}


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


    if arm == "oracle1":
        v = 1
        return SessionConfig(stream_task_ids=[], checkpoints=[0],
                             memory_mode="oracle", oracle_verbosity=v, **base), NoOpUpdater()
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


        return SessionConfig(stream_task_ids=[], checkpoints=[0],
                             memory_mode="oracle", oracle_verbosity=1,
                             inject_spend_totals=True, **base), NoOpUpdater()
    if arm == "oracle_autofill":


        return SessionConfig(stream_task_ids=[], checkpoints=[0],
                             memory_mode="oracle", oracle_verbosity=1,
                             autofill_spend_total=True, **base), NoOpUpdater()
    if arm == "learned":
        stream = build_stream(pool, args.n_train, args.seed)
        persona = Persona.load(persona_path)
        persona_rules = [r.name for r in persona.rules]
        distractors = [n for n in RULE_REGISTRY if n not in persona_rules][: args.distractors]


        bank = build_candidate_bank(persona_rules, distractors, shuffle_seed=args.seed)
        updater = AssertionTopL(bank, L=args.top_l)
        return SessionConfig(stream_task_ids=stream, checkpoints=args.checkpoints,
                             memory_mode="updater", **base), updater
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
    except json.JSONDecodeError:
        return False
    want_points = args.checkpoints if arm in STREAM_ARMS else [0]
    per_point = Counter(r["n"] for r in rows if r["phase"] == "eval")
    return all(per_point.get(n, 0) >= args.n_eval for n in want_points)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm", required=True)
    parser.add_argument("--persona", default="q1_format_checksum")
    parser.add_argument("--agent", default="fc", choices=["react", "fc", "oracle"],
                        help="fc = constrained function-call actuator; oracle = official solution")
    parser.add_argument("--arms", nargs="+", choices=sorted(STATIC_ARMS | STREAM_ARMS),
                        default=["baseline", "oracle1", "learned",
                                 "oracle_verifier", "oracle_verifier_value"])
    parser.add_argument("--n-train", type=int, default=8)
    parser.add_argument("--checkpoints", nargs="+", type=int, default=[0, 4, 8])
    parser.add_argument("--n-eval", type=int, default=7)
    parser.add_argument("--feedback-tier", default="corrective")
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--verifier-attempts", type=int, default=3)


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


    stem = f"exp1{args.tag}_{model_slug}_{args.persona}_s{args.seed}"
    out = OUTPUTS_DIR / f"{stem}.json"


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


        out.write_text(json.dumps(index, indent=1))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
