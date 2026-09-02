"""Merge per-model calibration jsons into the cross-model table (Assumption-1 flips).

Usage:
  python scripts/analyze_calibration.py outputs/calibration_*.json
"""

import argparse
import json
from pathlib import Path


def classify(oracle_compliance: float | None) -> str:
    if oracle_compliance is None:
        return "?"
    if oracle_compliance >= 0.95:
        return "in"
    if oracle_compliance < 0.70:
        return "OUT"
    return "border"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("calibrations", nargs="+")
    args = parser.parse_args()

    models: dict[str, dict] = {}
    rules: list[str] = []
    for path in args.calibrations:
        data = json.loads(Path(path).read_text())
        model = data["llm"].split(":")[-1]
        models[model] = data["results"]
        for r in data["results"]:
            if r not in rules:
                rules.append(r)

    names = list(models)
    header = f"{'rule':26s} {'pool':4s} " + " ".join(f"{m[:18]:>20s}" for m in names)
    print(header)
    print("-" * len(header))
    flips = []
    for rule in rules:
        cells, classes = [], []
        pool = "?"
        for m in names:
            res = models[m].get(rule)
            if not res:
                cells.append(f"{'—':>20s}")
                classes.append(None)
                continue
            pool = res["pool"]
            orc = res["arms"].get("oracle", {}).get("compliance")
            base = res["arms"].get("none", {}).get("compliance")
            anti = res["arms"].get("anti", {}).get("compliance")
            eta = None if anti is None else round(1 - anti, 2)
            cls = classify(orc)
            classes.append(cls)
            cells.append(f"{cls:>6s} o={orc if orc is not None else '—'}"
                         f" b={base if base is not None else '—'}"
                         f" e={eta if eta is not None else '—'}"[:20].rjust(20))
        seen = {c for c in classes if c and c != "?"}
        if len(seen - {"border"}) > 1:
            flips.append(rule)
        # disqualification flags — rules that cannot carry a clean calibration signal
        flags = []
        for m in names:
            res = models[m].get(rule) or {}
            none_arm = res.get("arms", {}).get("none", {})
            orc_arm  = res.get("arms", {}).get("oracle", {})
            if none_arm.get("compliance") is not None and none_arm["compliance"] >= 0.95:
                flags.append(f"base-saturated@{m.split(':')[-1][:8]}")
            if orc_arm.get("compliance_note"):
                flags.append(f"vacuous@{m.split(':')[-1][:8]}")
            if (orc_arm.get("applicable") or 0) < 3:
                flags.append(f"n<3@{m.split(':')[-1][:8]}")
        flag_str = "  !! " + ", ".join(flags) if flags else ""
        print(f"{rule:26s} {pool:4s} " + " ".join(cells) + flag_str)
    print(f"\nAssumption-1 flips (in-support on one model, OUT on another): {flips or 'none yet'}")
    print("!! flags mean the rule carries no usable calibration signal:")
    print("   base-saturated = empty-memory compliance >= 0.95, so oracle proves nothing")
    print("   vacuous        = every satisfaction held only because the precondition "
          "never fired")
    print("   n<3            = fewer than 3 applicable episodes; the rate is noise")


if __name__ == "__main__":
    main()
