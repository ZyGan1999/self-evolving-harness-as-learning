"""EpisodeRecord: everything checkers and updaters may need about one task episode."""

from dataclasses import dataclass, field
from typing import Any

from .apicalls import ApiCall


@dataclass
class EpisodeRecord:
    task_id: str
    instruction: str
    supervisor: dict[str, Any]          # first_name, last_name, email, phone_number
    api_calls: list[ApiCall]
    task_completed: bool                # agent called supervisor.complete_task
    tgc: bool | None = None             # official AppWorld evaluation success (None if not run)
    # Official TGC fails whenever a preference decorates a field AppWorld asserts exactly, so
    # it is not comparable across arms; tgc_relaxed excuses exactly that (appworld_p.tgc).
    tgc_relaxed: bool | None = None
    tgc_excused: list[str] = field(default_factory=list)  # requirements excused as decoration
    session_index: int = -1             # position in the interaction stream (-1 = eval episode)
    phase: str = "train"                # "train" (feedback flows) or "eval" (frozen c)
    meta: dict[str, Any] = field(default_factory=dict)
    # World-local lookup tables checkers need to speak a world-independent vocabulary.
    # card_names: payment_card_id -> bank name (each AppWorld world draws a random
    # 4-5 card subset of 7 banks, so raw ids are meaningless across worlds).
    card_names: dict[str, str] = field(default_factory=dict)
    # spend_totals: Venmo money already sent before this episode, as
    # {"total": float, "count": int, "per_recipient": {email: float}} -- the ground truth
    # for the exact-aggregation rules. Read from the world before the agent runs, so it
    # excludes whatever the agent is about to pay.
    spend_totals: dict[str, Any] = field(default_factory=dict)

    def calls(self, app: str, *apis: str) -> list[ApiCall]:
        """All resolved calls to app, optionally restricted to given api names."""
        return [c for c in self.api_calls
                if c.app == app and (not apis or c.api in apis)]
