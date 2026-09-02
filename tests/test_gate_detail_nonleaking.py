"""The reject-only gate must not name the answer.

This is the property the whole `checker rejects` column rests on. `usual_card`'s ordinary
`detail` says "first card was Chase, habitual card is American Express" -- feeding that into a
retry would turn the cell into a measurement of how fast f copies a leaked label, which is why
the arm was left out of the figure entirely before this channel existed. So the invariant gets a
test rather than a code comment: for every card the agent might have used, and every possible
target, the gate text must contain the card that WAS used and never the one that should be.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from appworld_p.apicalls import ApiCall  # noqa: E402
from appworld_p.episode import EpisodeRecord  # noqa: E402
from appworld_p.history import SessionHistory  # noqa: E402
from appworld_p.rules import RULE_REGISTRY  # noqa: E402

BANKS = ["American Express", "Wells Fargo", "Chase", "MasterCard", "HSBC", "Discover"]


def _episode(card_id: str, banks: dict[str, str]) -> EpisodeRecord:
    call = ApiCall(app="venmo", api="create_transaction", method="post",
                   arguments={"payment_card_id": card_id, "amount": 10,
                              "receiver_email": "x@y.com"})
    return EpisodeRecord(task_id="t", instruction="", supervisor={},
                         api_calls=[call], task_completed=True,
                         phase="eval", session_index=0, card_names=banks)


def test_gate_detail_never_names_the_target() -> None:
    rule = RULE_REGISTRY["usual_card"]()
    banks = {str(100 + i): b for i, b in enumerate(BANKS)}
    for target in BANKS:
        hist = SessionHistory()
        hist.habit_target = target
        for cid, used in banks.items():
            if used == target:
                continue  # not a violation, gate never fires
            text = rule.gate_detail(_episode(cid, banks), hist)
            assert used in text, f"gate must name the card actually used ({used}): {text!r}"
            assert target not in text, f"gate LEAKED the target {target!r}: {text!r}"
