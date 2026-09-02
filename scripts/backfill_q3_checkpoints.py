"""Backfill extra checkpoints into a finished Q3 self-evolution run, without re-running training.

The descent between n=0 and the first checkpoint was measured by two points, which cannot say
whether it is a step or a curve. Adding intermediate n is therefore about shape, and the shape has
to be read on ONE memory trajectory: re-running the stream to n=2 would resample the updater's
LLM calls at temperature 0.7 and produce a different memory than the run whose n=6 and n=12 we
already have, so the new point would not be comparable to its own neighbours.

So this script replays the finished run's recorded deltas to the requested stream position
(scripts/replay_ace_memory.py, verified exact against every dumped memory_n*.md on all three
seeds) and evaluates that memory with memory_mode="fixed" and eval_label_n set. Zero train
episodes are executed, every new point sits on the original trajectory, and the cost is only the
eval episodes.

Each backfilled checkpoint goes to its OWN session dir rather than appending to the original.
Appending would rewrite episodes.jsonl of a finished run, and a crash mid-write would corrupt
data that took 1.9h per seed to produce. The analysis scripts glob over session dirs and key on
each row's own n, so separate dirs merge into one curve automatically.

Usage:
  python scripts/backfill_q3_checkpoints.py --llm anthropic:claude-haiku-4-5-20251001 \
      --seeds 0 1 2 --at 1 2 3 4 --rollouts 3 [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from appworld_p.config import OUTPUTS_DIR, PERSONA_DIR, TASK_POOL_PATH  # noqa: E402
from appworld_p.driver import SessionConfig, SessionDriver  # noqa: E402
from appworld_p.updaters import NoOpUpdater  # noqa: E402
from replay_ace_memory import replay  # noqa: E402

SRC = "q3_p13_{slug}_{persona}_s{seed}_selfevolve_ace"
DST = "q3bf_{slug}_{persona}_s{seed}_ace_n{n}"


def _complete(run_name: str, want: int, label: int) -> bool:
    """Has this backfilled point already got all its eval episodes?"""
    path = OUTPUTS_DIR / "sessions" / run_name / "episodes.jsonl"
    if not path.exists():
        return False
    try:
        rows = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    except json.JSONDecodeError:
        return False
    return sum(1 for r in rows if r.get("phase") == "eval" and r.get("n") == label) >= want


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", required=True)
    ap.add_argument("--persona", default="p13_mixed")
    ap.add_argument("--agent", default="fc", choices=["react", "fc", "oracle"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--at", nargs="+", type=int, required=True,
                    help="stream positions to backfill, e.g. 1 2 3 4")
    ap.add_argument("--n-eval", type=int, default=6)
    ap.add_argument("--rollouts", type=int, default=3)
    ap.add_argument("--max-steps", type=int, default=50)
    ap.add_argument("--max-bullets", type=int, default=12)
    ap.add_argument("--skip-done", action="store_true")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the memory that would be injected at each n and stop")
    args = ap.parse_args()

    pool = json.loads(TASK_POOL_PATH.read_text())["pools"][args.persona]
    persona_path = str(PERSONA_DIR / f"{args.persona}.yaml")
    slug = args.llm.split(":")[-1].replace(".", "_")
    eval_ids = pool["eval_task_ids"][: args.n_eval]
    eval_ids = [t for t in eval_ids for _ in range(args.rollouts)]

    for seed in args.seeds:
        src = OUTPUTS_DIR / "sessions" / SRC.format(slug=slug, persona=args.persona, seed=seed)
        if not (src / "updater_final.json").exists():
            print(f"!! seed {seed}: no updater_final.json at {src}, skipping")
            continue
        mem = replay(src, args.max_bullets)
        for n in args.at:
            if n not in mem:
                print(f"!! seed {seed} n={n}: beyond the recorded stream, skipping")
                continue
            run_name = DST.format(slug=slug, persona=args.persona, seed=seed, n=n)
            block = mem[n]
            if args.dry_run:
                lines = [x for x in block.splitlines() if x.strip()]
                print(f"\n--- seed {seed} n={n} -> {run_name}  ({len(lines)} lines) ---")
                print(block)
                continue
            if args.skip_done and _complete(run_name, len(eval_ids), n):
                print(f"\n##### seed {seed} n={n} -> already complete, skipping")
                continue
            print(f"\n##### seed {seed} n={n} -> {run_name}")
            cfg = SessionConfig(
                run_name=run_name, persona_path=persona_path, stream_task_ids=[],
                eval_task_ids=eval_ids, checkpoints=[0], memory_mode="fixed",
                fixed_memory=block, eval_label_n=n, agent=args.agent, llm=args.llm,
                seed=seed, max_steps=args.max_steps, feedback_tier="instance",
                notes=f"q3 backfill: ACE memory replayed to stream position n={n}")
            manifest = SessionDriver(cfg, NoOpUpdater()).run()
            print(f"  eval_points: {manifest['eval_points']}")


if __name__ == "__main__":
    main()
