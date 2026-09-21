"""Run the reported Q2 R-arm memory-length sweep."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from appworld_p.arms import build_arm_memory  # noqa: E402
from appworld_p.config import OUTPUTS_DIR, PERSONA_DIR, TASK_POOL_PATH  # noqa: E402
from appworld_p.driver import SessionConfig, SessionDriver  # noqa: E402
from appworld_p.summarize import Assertion  # noqa: E402
from appworld_p.updaters import NoOpUpdater  # noqa: E402


def load_pool(tag: str, model_slug: str, persona: str, seed: int) -> list:
    path = OUTPUTS_DIR / f"exp2pool{tag}_{model_slug}_{persona}_s{seed}.json"
    return [Assertion(**a) for a in json.loads(path.read_text())["assertions"]]


def score_run(run_name: str, rule_names: list[str]) -> dict:
    path = OUTPUTS_DIR / "sessions" / run_name / "episodes.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    rows = [row for row in rows if row.get("phase") == "eval"]
    out = {}
    for name in rule_names:
        applicable = [row["rules"][name] for row in rows
                      if row.get("rules", {}).get(name, {}).get("applicable")]
        violated = sum(res["satisfied"] is False for res in applicable)
        n = len(applicable)
        out[name] = {"episodes": len(rows), "applicable": n, "violated": violated,
                     "violation_rate": violated / n if n else None}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", required=True)
    ap.add_argument("--persona", default="p8_q2s1")
    ap.add_argument("--arms", nargs="+", default=["R"], choices=["R"])
    ap.add_argument("--l-values", nargs="+", type=int,
                    default=[0, 1, 2, 3, 5, 10, 20, 60, 150])
    ap.add_argument("--pick", default="consensus", choices=["consensus", "earliest"])
    ap.add_argument("--cap-distinct", type=int, default=55,
                    help="cap on distinct additional assertion texts before recycling")
    ap.add_argument("--n-eval", type=int, default=12)
    ap.add_argument("--max-steps", type=int, default=50)
    ap.add_argument("--agent", default="fc", choices=["fc", "react"])
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--pool-seeds", nargs="+", type=int, default=[1, 2])
    ap.add_argument("--rollouts", type=int, default=10)
    ap.add_argument("--tag", default="_v1")
    ap.add_argument("--out-suffix", default="")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if not args.l_values or min(args.l_values) < 0 or args.rollouts < 1:
        ap.error("memory lengths must be nonnegative and rollouts must be positive")

    slug = args.llm.split(":")[-1].replace(".", "_")
    scored = [a for seed in args.pool_seeds
              for a in load_pool(args.tag, slug, args.persona, seed)]
    pool = json.loads(TASK_POOL_PATH.read_text())["pools"][args.persona]
    eval_ids = [task for task in pool["eval_task_ids"][:args.n_eval]
                for _ in range(args.rollouts)]
    persona_rules = list(pool["eval_rule_cover"])
    d = len(persona_rules)
    grid = []
    for L in args.l_values:
        memory, diag = build_arm_memory(
            "R", scored, d, L, pick=args.pick, seed=args.seed,
            cap_distinct=args.cap_distinct)
        print(f"R L={L}: {diag['lines']} lines, {diag['rules_covered']} rules, "
              f"{len(eval_ids)} evaluation episodes", flush=True)
        if args.dry_run:
            continue
        run = (f"exp2arms{args.tag}{args.out_suffix}_{slug}_{args.persona}"
               f"_R_L{L}_s{args.seed}")
        cfg = SessionConfig(
            run_name=run, persona_path=str(PERSONA_DIR / f"{args.persona}.yaml"),
            stream_task_ids=[], eval_task_ids=eval_ids, checkpoints=[0],
            agent=args.agent, llm=args.llm, seed=args.seed,
            max_steps=args.max_steps, memory_mode="fixed", fixed_memory=memory,
            notes=f"exp2 arm=R L={L}",
        )
        manifest = SessionDriver(cfg, NoOpUpdater()).run()
        grid.append({**diag, "memory_chars": len(memory),
                     **manifest["eval_points"]["0"],
                     "per_rule": score_run(run, persona_rules)})
        out = OUTPUTS_DIR / (f"exp2armsgrid{args.tag}{args.out_suffix}_{slug}"
                             f"_{args.persona}_s{args.seed}.json")
        out.write_text(json.dumps({"args": vars(args), "grid": grid}, indent=1))


if __name__ == "__main__":
    main()
