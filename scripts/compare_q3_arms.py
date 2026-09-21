"""Aggregate Q3 evaluation episodes by method, seed, and checkpoint."""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from appworld_p.config import OUTPUTS_DIR  # noqa: E402

MODEL = "claude-haiku-4-5-20251001"
PERSONA = "p13_mixed"
SEEDS = (0, 1, 2)
NS = [0, 6, 12, 18, 24]


RULES = ("sms_greeting", "sms_signoff", "payment_note_category",
         "payment_note_initials", "venmo_private")


ARMS = {
    "ACE": f"q3_p13_{MODEL}_{PERSONA}_s{{s}}_selfevolve_ace",
    "TEPA": f"q3_p13bl_{MODEL}_{PERSONA}_s{{s}}_tepa",
    "TRACE": f"q3_p13bl_{MODEL}_{PERSONA}_s{{s}}_trace",
    "Reflexion": f"q3_p13bl_{MODEL}_{PERSONA}_s{{s}}_reflexion",
}

REFS = {"none": f"q3_p13_{MODEL}_{PERSONA}_s0_none",
        "oracle": f"q3_p13_{MODEL}_{PERSONA}_s0_oracle"}


def counts(dirname: str) -> dict[int, list[int]] | None:
    """-> {n: [applicable, violated]} for one run, or None if the run is absent."""
    path = OUTPUTS_DIR / "sessions" / dirname / "episodes.jsonl"
    if not path.exists():
        return None
    cell: dict[int, list[int]] = defaultdict(lambda: [0, 0])
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("phase") != "eval":
            continue
        for name, res in (row.get("rules") or {}).items():
            if name in RULES and res.get("applicable"):
                cell[row["n"]][0] += 1
                cell[row["n"]][1] += int(res["satisfied"] is False)
    return dict(cell)


def collect() -> dict:
    out = {"arms": {}, "refs": {}, "diagnostics": {}}
    for arm, pattern in ARMS.items():
        out["arms"][arm] = {}
        for s in SEEDS:
            c = counts(pattern.format(s=s))
            if c is None:
                continue
            out["arms"][arm][s] = {n: {"applicable": a, "violated": v, "rate": v / a}
                                  for n, (a, v) in sorted(c.items()) if a}
    for label, dirname in REFS.items():
        c = counts(dirname)
        if c and 0 in c:
            a, v = c[0]
            out["refs"][label] = {"applicable": a, "violated": v, "rate": v / a}
    for label, arm in [("Full rewrite", "selfevolve_rewrite"),
                       ("Corrective", "external_corrective")]:
        c = counts(f"q3_p13_{MODEL}_{PERSONA}_s0_{arm}")
        if c:
            n = max(c)
            a, v = c[n]
            out["diagnostics"][label] = {"checkpoint": n, "applicable": a,
                                         "violated": v, "rate": v / a if a else None}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true", help="dump the aggregation instead of a table")
    args = ap.parse_args()
    data = collect()
    if args.json:
        print(json.dumps(data, indent=1))
        return

    print(f"{'arm/seed':<14}" + "".join(f"{'n='+str(n):>9}" for n in NS) + f"{'mean n=24':>11}")
    print("-" * 76)
    for arm, seeds in data["arms"].items():
        finals = []
        for s, points in seeds.items():
            row = ""
            for n in NS:
                row += f"{points[n]['rate']:>9.3f}" if n in points else f"{'--':>9}"
            if NS[-1] in points:
                finals.append(points[NS[-1]]["rate"])
            print(f"{arm + ' s' + str(s):<14}{row}")
        if finals:
            print(f"{'':<14}{'':>45}{sum(finals) / len(finals):>11.3f}")
    print("-" * 76)
    for label, ref in data["refs"].items():
        print(f"{label:<14}{ref['rate']:>9.3f}   ({ref['violated']}/{ref['applicable']})")

    for label, point in data["diagnostics"].items():
        rate = point["rate"]
        rendered = f"{rate:.3f}" if rate is not None else "--"
        print(f"{label}: n={point['checkpoint']} "
              f"{point['violated']}/{point['applicable']} = {rendered}")


if __name__ == "__main__":
    main()
