"""EpisodeRecord: everything checkers and updaters may need about one task episode."""

from dataclasses import dataclass, field
from typing import Any

from .apicalls import ApiCall


@dataclass
class EpisodeRecord:
    task_id: str
    instruction: str
    supervisor: dict[str, Any]
    api_calls: list[ApiCall]
    task_completed: bool
    tgc: bool | None = None

    tgc_relaxed: bool | None = None
    tgc_excused: list[str] = field(default_factory=list)
    session_index: int = -1
    phase: str = "train"
    meta: dict[str, Any] = field(default_factory=dict)

    card_names: dict[str, str] = field(default_factory=dict)

    spend_totals: dict[str, Any] = field(default_factory=dict)

    def calls(self, app: str, *apis: str) -> list[ApiCall]:
        """All resolved calls to app, optionally restricted to given api names."""
        return [c for c in self.api_calls
                if c.app == app and (not apis or c.api in apis)]
