"""R002: build task_pool.json from train+dev metadata.

For each persona: filter tasks by difficulty and app overlap with the persona's
rules, then greedily pick a frozen eval set covering each relevant app enough
times; the rest becomes the interaction-stream pool.

Usage: conda run -n appworld-p python scripts/build_task_pool.py [--min-cover 5] [--eval-size 25]
"""

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from appworld_p.config import PERSONA_DIR, TASK_POOL_PATH, set_appworld_root  # noqa: E402
from appworld_p.persona import Persona  # noqa: E402


def load_task_metadata() -> list[dict]:
    set_appworld_root()
    from appworld import AppWorld, load_task_ids

    tasks = []
    for split in ("train", "dev"):
        for task_id in load_task_ids(split):
            with AppWorld(task_id=task_id, experiment_name="pool_scan",
                          ground_truth_mode="full", raise_on_failure=False) as world:
                gt = world.task.ground_truth
                meta = gt.metadata if hasattr(gt, "metadata") else {}
                tasks.append({
                    "task_id": task_id,
                    "split": split,
                    "difficulty": meta.get("difficulty"),
                    "num_apis": meta.get("num_apis"),
                    "num_apps": meta.get("num_apps"),
                    "required_apps": sorted(getattr(gt, "required_apps", []) or []),
                    "required_apis": sorted(getattr(gt, "required_apis", []) or []),
                    "instruction": world.task.instruction,
                })
    return tasks


def build_pool_for_persona(persona: Persona, tasks: list[dict], max_difficulty: int,
                           eval_size: int, min_cover: int, rng: random.Random) -> dict:
    """Eligibility = the official solution invokes at least one rule-triggering API
    (required_apis vs rule.trigger_apis), not mere app overlap — this guarantees
    every stream/eval task actually exercises some preference rule."""
    eligible = []
    for t in tasks:
        if (t["difficulty"] or 99) > max_difficulty:
            continue
        triggered = persona.rules_triggered_by(set(t.get("required_apis", [])))
        if triggered:
            eligible.append({**t, "triggered_rules": triggered})
    rng.shuffle(eligible)
    # never let the eval set eat more than half the (small) eligible pool
    eval_size = min(eval_size, max(len(eligible) // 2, 1))

    # greedy eval-set cover: each rule triggered >= min_cover times if possible
    eval_set: list[dict] = []
    cover: Counter = Counter()

    # AppWorld task ids are <template>_<variant>: '530b157_1/_2/_3' are three variants of
    # ONE template. Covering a rule 4x from a single template measures that template's
    # quirks, not the rule -- p4_q1's first build put 3 of 4 sms tasks in one family. Break
    # ties toward the least-used template so eval spreads across templates.
    fam_used: Counter = Counter()

    def gain(t):
        want = sum(1 for r in t["triggered_rules"] if cover[r] < min_cover)
        # rank: covers anything new > least-used template > covers the most. Family
        # diversity must outrank the count, or a template that happens to trigger both
        # rules wins every pick and fills eval by itself (it did: 530b157 took 3 of 4).
        return (min(want, 1), -fam_used[t["task_id"].split("_")[0]], want)

    remaining = list(eligible)
    while remaining and len(eval_set) < eval_size:
        best = max(remaining, key=gain)
        if gain(best)[0] == 0 and len(eval_set) >= min(eval_size, 10):
            break
        fam_used[best["task_id"].split("_")[0]] += 1
        eval_set.append(best)
        remaining.remove(best)
        for r in best["triggered_rules"]:
            cover[r] += 1

    stream_pool = remaining
    rule_names = [r.name for r in persona.rules]
    return {
        "persona": persona.name,
        "eligible_count": len(eligible),
        "eval_task_ids": [t["task_id"] for t in eval_set],
        "stream_task_ids": [t["task_id"] for t in stream_pool],
        "eval_rule_cover": {r: cover.get(r, 0) for r in rule_names},
        "uncovered_rules": [r for r in rule_names if cover.get(r, 0) == 0],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-difficulty", type=int, default=2)
    parser.add_argument("--eval-size", type=int, default=25)
    parser.add_argument("--min-cover", type=int, default=5)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--from-cache", action="store_true",
                        help="reuse task_metadata from the existing task_pool.json "
                             "instead of rescanning every world (~30 min)")
    parser.add_argument("--only", nargs="+", default=None,
                        help="build pools only for these personas; existing pools are "
                             "kept byte-identical (frozen eval sets must not move)")
    args = parser.parse_args()

    existing = (json.loads(TASK_POOL_PATH.read_text())
                if TASK_POOL_PATH.exists() and args.from_cache else None)
    if existing:
        tasks = existing["task_metadata"]
        print(f"  reusing cached metadata for {len(tasks)} tasks")
    else:
        print("Scanning train+dev task metadata (first run loads all apps, ~min)...")
        tasks = load_task_metadata()
        print(f"  {len(tasks)} tasks scanned")

    pools = dict(existing["pools"]) if existing else {}
    for yaml_path in sorted(PERSONA_DIR.glob("*.yaml")):
        persona = Persona.load(yaml_path)
        if args.only and persona.name not in args.only:
            continue
        # per-persona rng: adding a persona must not perturb any other pool
        rng = random.Random(f"{args.seed}:{persona.name}")
        pool = build_pool_for_persona(persona, tasks, args.max_difficulty,
                                      args.eval_size, args.min_cover, rng)
        pools[persona.name] = pool
        print(f"  {persona.name}: eligible={pool['eligible_count']} "
              f"eval={len(pool['eval_task_ids'])} stream={len(pool['stream_task_ids'])} "
              f"rule_cover={pool['eval_rule_cover']} uncovered={pool['uncovered_rules']}")

    TASK_POOL_PATH.parent.mkdir(parents=True, exist_ok=True)
    config = {**(existing.get("config", {}) if existing else {}), **vars(args)}
    TASK_POOL_PATH.write_text(json.dumps(
        {"config": config, "tasks_scanned": len(tasks), "pools": pools,
         "task_metadata": tasks}, indent=1))
    print(f"Wrote {TASK_POOL_PATH}")


if __name__ == "__main__":
    main()
