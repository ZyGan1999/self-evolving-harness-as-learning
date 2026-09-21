"""Aggregate the reported Q2 violation rates across seeds and memory lengths."""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from appworld_p.config import OUTPUTS_DIR  # noqa: E402

RUN = "exp2arms_v1_claude-haiku-4-5-20251001_p8_q2s1_R_L{L}_s{s}"
RULES = ["sms_signoff", "sms_greeting", "playlist_over_like", "venmo_private",
         "payment_note_single_word", "payment_note_lowercase",
         "payment_has_note", "sms_terse"]
SEEDS = (1, 2, 3)
BRANCH = (0, 1, 2, 3, 5, 10, 20, 60, 150)

def read(L, s):
    """{(rule, task): [viol, applicable]} or None if that cell has not been written yet."""
    p = OUTPUTS_DIR / "sessions" / RUN.format(L=L, s=s) / "episodes.jsonl"
    if not p.exists():
        return None
    out = defaultdict(lambda: [0, 0])
    for line in p.read_text().splitlines():
        if not line.strip():
            continue
        ep = json.loads(line)
        if ep.get("phase") != "eval":
            continue
        for name, r in (ep.get("rules") or {}).items():
            if name in RULES and r.get("applicable"):
                out[(name, ep["task_id"])][0] += r.get("satisfied") is False
                out[(name, ep["task_id"])][1] += 1
    return out

def tot(cell, rules):
    if cell is None:
        return 0, 0
    rs = rules if isinstance(rules, (list, tuple, set)) else (rules,)
    return (sum(v[0] for k, v in cell.items() if k[0] in rs),
            sum(v[1] for k, v in cell.items() if k[0] in rs))

def wilson(v, n, z=1.96):
    if not n:
        return float("nan"), float("nan"), float("nan")
    p = v / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / den
    return p, max(0.0, c - h), min(1.0, c + h)

def collect(excluded=()):
    keep = [rule for rule in RULES if rule not in excluded]
    result = {"rules": keep, "per_seed": {}, "pooled": {}}
    for seed in SEEDS:
        points = {}
        for length in BRANCH:
            cell = read(length, seed)
            if cell is None:
                continue
            violated, applicable = tot(cell, keep)
            rate, low, high = wilson(violated, applicable)
            points[length] = {
                "violated": violated, "applicable": applicable,
                "rate": rate if applicable else None,
                "wilson95": [low, high] if applicable else None,
            }
        result["per_seed"][seed] = points
    for length in BRANCH:
        cells = [points[length] for points in result["per_seed"].values()
                 if length in points]
        violated = sum(c["violated"] for c in cells)
        applicable = sum(c["applicable"] for c in cells)
        rate, low, high = wilson(violated, applicable)
        result["pooled"][length] = {
            "seeds_present": len(cells),
            "complete": len(cells) == len(SEEDS),
            "violated": violated, "applicable": applicable,
            "rate": rate if applicable else None,
            "wilson95": [low, high] if applicable else None,
        }
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exclude", nargs="*", default=[], choices=RULES,
                    help="omit rules from scoring the same eight-rule-harness episodes")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    data = collect(args.exclude)
    if args.json:
        print(json.dumps(data, indent=2, allow_nan=False))
        return
    print("L   seed 1   seed 2   seed 3   pooled violations/applicable")
    for length in BRANCH:
        rates = [data["per_seed"][seed].get(length, {}).get("rate") for seed in SEEDS]
        cells = " ".join(f"{r:8.3f}" if r is not None else "      --" for r in rates)
        pooled = data["pooled"][length]
        suffix = "" if pooled["complete"] else " (incomplete)"
        print(f"{length:<3}{cells}   {pooled['violated']}/{pooled['applicable']}{suffix}")


if __name__ == "__main__":
    main()
