"""Run the Q1 habitual-card comparison.
Context arms use no memory, a stated rule, or the full observation log.
Control arms use an external statistic or rejection-guided retries.
The schedule counts synthetic habit observations."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from appworld_p.config import OUTPUTS_DIR, PERSONA_DIR  # noqa: E402
from appworld_p.driver import SessionConfig, SessionDriver  # noqa: E402
from appworld_p.habit import FAMILY_B_EVAL_TASKS, FAMILY_B_OPTIONS  # noqa: E402
from appworld_p.updaters import NoOpUpdater  # noqa: E402

ARMS = ("baseline", "oracle", "fulllog", "external", "gate")


def arm_config(arm: str, run_name: str, persona_path: str,
               args: argparse.Namespace) -> SessionConfig:


    schedule = args.schedule if arm in ("fulllog", "external") else [args.schedule[-1]]
    base = dict(run_name=run_name, persona_path=persona_path, agent=args.agent,
                llm=args.llm, seed=args.seed, max_steps=args.max_steps,
                eval_task_ids=list(FAMILY_B_EVAL_TASKS)[: args.n_eval],
                stream_task_ids=[], checkpoints=[],
                habit_dominant=args.dominant, habit_tier=args.tier,
                habit_noisy=args.noisy, habit_noise_per_round=args.noise_density,
                habit_schedule=schedule,
                notes=f"exp1b arm={arm} tier={args.tier} dominant={args.dominant}")
    if arm == "baseline":
        return SessionConfig(memory_mode="none", **base)
    if arm == "oracle":
        return SessionConfig(memory_mode="oracle", oracle_verbosity=1, **base)
    if arm == "fulllog":
        return SessionConfig(memory_mode="fulllog", oracle_verbosity=1, **base)
    if arm == "external":
        return SessionConfig(memory_mode="oracle", oracle_verbosity=1,
                             inject_stats=True, **base)
    if arm == "gate":


        return SessionConfig(memory_mode="oracle", oracle_verbosity=1,
                             verifier_attempts=args.attempts, verifier_nonleaking=True,
                             verifier_accumulate=True, **base)
    raise ValueError(f"unknown arm: {arm}")


def summarize(run_name: str) -> dict:
    """Per-checkpoint habitual-card violations over applicable evaluation episodes."""
    path = OUTPUTS_DIR / "sessions" / run_name / "episodes.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    by_n: dict[str, dict] = {}
    for row in rows:
        if row.get("phase") != "eval":
            continue
        slot = by_n.setdefault(str(row["n"]), {
            "episodes": 0, "applicable": 0, "violations": 0,
        })
        slot["episodes"] += 1
        rule = row["rules"].get("usual_card", {})
        if rule.get("applicable"):
            slot["applicable"] += 1
            slot["violations"] += int(rule.get("satisfied") is False)
    for slot in by_n.values():
        slot["violation_rate"] = (
            slot["violations"] / slot["applicable"] if slot["applicable"] else None
        )
    return by_n


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm", required=True)
    parser.add_argument("--persona", default="q1_habitual_card")
    parser.add_argument("--agent", default="fc", choices=["fc", "react"],
                        help="Agent execution interface.")
    parser.add_argument("--arms", nargs="+", default=list(ARMS))
    parser.add_argument("--schedule", nargs="+", type=int, default=[0, 20, 60, 120],
                        help="cumulative habit observations to evaluate at")
    parser.add_argument("--dominant", default=FAMILY_B_OPTIONS[0], choices=FAMILY_B_OPTIONS,
                        help="Card brand with the highest probability in the latent preference distribution.")
    parser.add_argument("--attempts", type=int, default=3,
                        help="Maximum attempts per episode for the gate arm.")
    parser.add_argument("--tier", default="binary", choices=["binary", "corrective"],
                        help="Feedback format for synthetic habit observations.")
    parser.add_argument("--noisy", action="store_true",
                        help="dilute the log with unrelated interactions")
    parser.add_argument("--noise-density", type=float, default=1.0,
                        help="mean unrelated lines per habit round (with --noisy)")
    parser.add_argument("--n-eval", type=int, default=6)
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tag", default="")
    args = parser.parse_args()

    persona_path = str(PERSONA_DIR / f"{args.persona}.yaml")
    model_slug = args.llm.split(":")[-1].replace(".", "_")
    noise = f"_d{args.noise_density:g}" if args.noisy else ""
    slug = (f"exp1b{args.tag}_{model_slug}_{args.dominant.split()[0].lower()}"
            f"_{args.tier}{noise}_s{args.seed}")

    index = {"llm": args.llm, "persona": args.persona, "args": vars(args), "arms": {}}
    for arm in args.arms:
        run_name = f"{slug}_{arm}"
        print(f"\n##### arm {arm} -> {run_name}", flush=True)
        config = arm_config(arm, run_name, persona_path, args)
        manifest = SessionDriver(config, NoOpUpdater()).run()
        index["arms"][arm] = {"run_name": run_name, "habit": manifest["habit"],
                              "by_n": summarize(run_name),
                              "llm_usage": manifest["llm_usage"]}
        print(f"  {arm} by_n: " + json.dumps(index['arms'][arm]['by_n']), flush=True)

    out = OUTPUTS_DIR / f"{slug}.json"
    out.write_text(json.dumps(index, indent=1))

    print(f"\n=== exp1b: target={args.dominant} tier={args.tier} seed={args.seed} ===")
    ns = [str(n) for n in args.schedule]
    print("Habitual-card violation rate (violations/applicable)")
    print(f"{'arm':10s} " + " ".join(f"{'n=' + n:>18s}" for n in ns))
    for arm, res in index["arms"].items():
        cells = []
        for n in ns:
            point = res["by_n"].get(n)
            if point is None or point["violation_rate"] is None:
                cells.append(f"{'--':>18s}")
            else:
                cell = f"{point['violation_rate']:.3f} ({point['violations']}/{point['applicable']})"
                cells.append(f"{cell:>18s}")
        print(f"{arm:10s} " + " ".join(cells))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
