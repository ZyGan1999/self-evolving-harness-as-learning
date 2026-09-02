"""Classify each out-of-support preference as context-INEFFICIENT or context-INSUFFICIENT.

The body figure labels the three out-of-support panels with one of two subtypes, and the label
changes what the panel claims, so it is measured here rather than asserted in a caption.

The test: is the preference's target a STATISTIC OF THE HISTORY? If it is, a long enough log
recovers it and the harness's contribution is efficiency. If it is not, no amount of log helps and
the harness has to supply the computation. Two independent readings, because the checkpoint sweep
alone is weak evidence -- the exp1 arms only reach n=8, and a flat curve over a short range could
just be a curve that has not started moving:

  1. TREND. Violation rate over the history sweep of the strongest context arm. A monotone
     decline says the log is being mined successfully.

  2. CEILING. What happens when the external history is handed over complete, as a computed
     statistic (oracle_stats), instead of as raw log the model must traverse. This is the external
     history AT ITS LIMIT: if a preference is still violated when the aggregate is stated
     outright, the residual cannot be a history-access problem, whatever the trend looks like.

Usage: python scripts/context_subtype.py
"""
from __future__ import annotations

import glob
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from appworld_p.config import OUTPUTS_DIR  # noqa: E402

# rule -> (session glob for its strongest context arm, label)
TREND_ARM = {
    "sms_char_checksum": ("exp1_q1main_*_learned", "learned"),
    "spend_total_per_recipient": ("exp1_q1cross_*_learned", "learned"),
    "usual_card": ("exp1b_m1_*_fulllog", "fulllog"),
}
CEILING_ARM = {"spend_total_per_recipient": "exp1_q1cross_*_oracle_stats"}


def sweep(pattern: str, rule: str) -> dict[int, tuple[int, int]]:
    """violation counts per checkpoint, pooled over seeds/rotations."""
    agg: dict[int, list[int]] = defaultdict(lambda: [0, 0])
    for root in ("sessions", "sessions_server"):
        for path in sorted(glob.glob(str(OUTPUTS_DIR / root / pattern / "episodes.jsonl"))):
            for line in Path(path).read_text().splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                verdict = row.get("rules", {}).get(rule, {})
                if row["phase"] != "eval" or not verdict.get("applicable"):
                    continue
                agg[row["n"]][1] += 1
                agg[row["n"]][0] += int(verdict["satisfied"] is False)
    return {n: tuple(v) for n, v in sorted(agg.items())}


def classify(rule: str) -> tuple[str, str]:
    """-> (subtype, one-line evidence)."""
    pattern, arm = TREND_ARM[rule]
    curve = sweep(pattern, rule)
    rates = [v / a for v, a in curve.values()]
    shown = "  ".join(f"n={n}:{v}/{a}={v/a:.2f}" for n, (v, a) in curve.items())

    if rule in CEILING_ARM:
        c = sweep(CEILING_ARM[rule], rule)
        v, a = (sum(x[0] for x in c.values()), sum(x[1] for x in c.values()))
        if a and v / a > 0.05:
            # decisive on its own: the aggregate was stated outright and it still failed
            return ("context is insufficient",
                    f"complete ledger injected -> {v}/{a}={v/a:.2f}; {arm} sweep {shown}")

    # no ceiling probe for this rule: fall back to the trend. Require a real decline, not just
    # any downward step, and require the endpoint to be well clear of the start.
    declining = all(b <= a + 1e-9 for a, b in zip(rates, rates[1:]))
    moved = len(rates) > 1 and (rates[0] - rates[-1]) > 0.25
    if declining and moved:
        return ("context is inefficient", f"{arm} declines monotonically: {shown}")
    return ("context is insufficient", f"{arm} flattens: {shown}")


def main() -> None:
    print("out-of-support subtype (measured, not asserted)\n")
    for rule in TREND_ARM:
        subtype, why = classify(rule)
        print(f"  {rule:28s} {subtype}")
        print(f"  {'':28s}   {why}\n")


if __name__ == "__main__":
    main()
