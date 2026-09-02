"""Cross-session interaction history.

This is *driver-side* state: the ground truth that pool-B (state-class) preference
rules are defined against, and the data source for the external-counter control
harness. It is NOT visible to the agent unless a harness explicitly injects it.
"""

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from .episode import EpisodeRecord

# Logical payment actions we track for frequency/recency rules.
PAYMENT_APPS = ("venmo", "splitwise")


@dataclass
class SessionHistory:
    payment_card_choices: list[str] = field(default_factory=list)  # card ids (venmo/amazon), in order
    payment_count: int = 0                                         # money-transfer txns across sessions
    email_count: int = 0
    episodes_seen: int = 0
    # --- family B (statistical aggregation): driver-side habit ground truth ---
    # Set by the driver from its HabitStream; the checker for `usual_card` reads it.
    # Never exposed to the agent (arms see the log or the counter, never this).
    habit_target: str | None = None      # bank name of the user's habitual card
    habit_rounds: int = 0                # how many habit observations exist so far

    def update(self, ep: EpisodeRecord) -> None:
        self.episodes_seen += 1
        for call in ep.api_calls:
            if call.app in PAYMENT_APPS and call.method == "post" and "transaction" in call.url:
                self.payment_count += 1
                card = call.arg("payment_card_id", "card_id")
                if card is not None:
                    self.payment_card_choices.append(str(card))
            if call.app == "amazon" and call.method == "post" and "order" in call.url:
                card = call.arg("payment_card_id", "card_id")
                if card is not None:
                    self.payment_card_choices.append(str(card))
            if call.app == "gmail" and call.method == "post" and ("email" in call.url or "thread" in call.url):
                self.email_count += 1

    def recent_mode_card(self, window: int = 5) -> str | None:
        recent = self.payment_card_choices[-window:]
        if not recent:
            return None
        return Counter(recent).most_common(1)[0][0]

    def render_external_stats(self) -> str:
        """Structured summary injected by the external-counter control harness.
        The decision maker is still f — but the state lives outside it."""
        lines = [
            f"Total payments made for this user so far: {self.payment_count} "
            f"(the next one is payment #{self.payment_count + 1}).",
        ]
        recent = self.payment_card_choices[-5:]
        if recent:
            counts = ", ".join(f"card {c}: {n}x" for c, n in Counter(recent).most_common())
            lines.append(f"Payment cards used in the last {len(recent)} card payments: {counts}. "
                         f"Most used: card {self.recent_mode_card()}.")
        lines.append(f"Emails sent so far: {self.email_count}. "
                     f"Episodes interacted: {self.episodes_seen}.")
        return "\n".join(lines)

    # -- persistence -------------------------------------------------------
    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.__dict__, indent=1))

    @classmethod
    def load(cls, path: str | Path) -> "SessionHistory":
        p = Path(path)
        if not p.exists():
            return cls()
        return cls(**json.loads(p.read_text()))
