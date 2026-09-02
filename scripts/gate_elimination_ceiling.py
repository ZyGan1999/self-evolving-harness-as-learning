"""What a card-cycling strategy scores on the reject-only gate, with no preference knowledge.

The reject-only arm hands back one bit per attempt ("not that card"). Over a attempts that is a
channel, and on a small option set the channel alone can find the target -- so a 0.00 in that
cell would say nothing about learning the habit. This script computes the ceiling that channel
gives an agent that ignores the preference entirely and just cycles through distinct cards.

Read the arm against this number, not against zero. If the measured violation rate sits at the
ceiling, the cell is a design artifact; only a rate clearly BELOW it is evidence the mechanism
did something. The card counts are read from the eval worlds rather than assumed: AppWorld draws
a random 4-5 card subset of 7 banks per world (see EpisodeRecord.card_names), which is why the
effective option set is not the 3 banks the habit stream rotates over.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from appworld_p.config import OUTPUTS_DIR  # noqa: E402

# Measured with scripts/probe_family_b_cards.py against the six family-B eval worlds.
CARDS_PER_WORLD = {"37a8675_2": 4, "37a8675_3": 5, "530b157_2": 5,
                   "60d0b5b_1": 5, "60d0b5b_2": 4, "60d0b5b_3": 5}


def ceiling(k: int, attempts: int) -> float:
    """Violation rate for uniform sampling WITHOUT replacement over k cards in `attempts` tries.

    Without replacement is the right model because the gate names the card just used and the
    driver accumulates exclusions, so a competent agent never repeats one. That makes this an
    upper bound on the channel's power, which is what we want to rule out.
    """
    return 0.0 if attempts >= k else (k - attempts) / k


def pooled_ceiling(attempts: int, tasks=None) -> float:
    """Ceiling over the eval set actually used, weighting each world equally."""
    ks = [CARDS_PER_WORLD[t] for t in (tasks or CARDS_PER_WORLD)]
    return sum(ceiling(k, attempts) for k in ks) / len(ks)


def main() -> None:
    print("排除法天花板（违规率下限；纯轮换、不看历史）")
    print(f"{'attempts':>9s}  " + "  ".join(f"k={k}" for k in (3, 4, 5)) + "   pooled(eval set)")
    for a in (1, 2, 3, 4):
        cells = "  ".join(f"{ceiling(k, a):.2f}" for k in (3, 4, 5))
        print(f"{a:>9d}  {cells}   {pooled_ceiling(a):.2f}")
    print()
    print("评测世界的卡数:", CARDS_PER_WORLD)
    print("习惯流只轮换 3 个品牌，但 agent 面对的是世界里的全部卡，")
    print("所以有效候选集是 4-5 而非 3 —— 这是 attempts=3 仍不能被排除法做穿的原因。")
    out = OUTPUTS_DIR / "gate_elimination_ceiling.json"
    out.write_text(json.dumps(
        {"cards_per_world": CARDS_PER_WORLD,
         "ceiling_by_attempts": {a: pooled_ceiling(a) for a in (1, 2, 3, 4)}}, indent=1))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
