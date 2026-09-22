"""Preference rule framework.

A Rule is the unit of f*_u: (hidden semantics, programmatic checker, applicability
predicate).

Checkers read EpisodeRecord (resolved api_calls) + SessionHistory. Duplicate field-claim keys are rejected when constructing a persona.
"""

import re
from dataclasses import dataclass
from typing import Callable

from ..episode import EpisodeRecord
from ..history import SessionHistory


@dataclass
class RuleResult:
    rule: str
    applicable: bool
    satisfied: bool | None
    detail: str = ""


class Rule:
    name: str = ""
    pool: str = ""
    apps: tuple[str, ...] = ()
    trigger_actions: tuple[str, ...] = ()
    field_claims: tuple[str, ...] = ()
    oracle_text: str = ""
    negation_text: str = ""

    correction_template: str = ""

    def trigger_apis(self) -> set[str]:
        """'{app}.{api}' strings that trigger this rule (matches ground_truth.required_apis)."""
        from .api_map import ACTIONS
        return {f"{ACTIONS[a].app}.{api}"
                for a in self.trigger_actions for api in ACTIONS[a].api_candidates}

    def gate_detail(self, ep: EpisodeRecord, history: SessionHistory) -> str:
        """Return rejection details without revealing the computed target value."""
        return ""

    def applicable(self, ep: EpisodeRecord) -> bool:
        raise NotImplementedError

    def satisfied(self, ep: EpisodeRecord, history: SessionHistory) -> tuple[bool, str]:
        """Only called when applicable. Returns (ok, detail).
        """
        raise NotImplementedError

    def check(self, ep: EpisodeRecord, history: SessionHistory) -> RuleResult:
        if not self.applicable(ep):
            return RuleResult(self.name, applicable=False, satisfied=None)
        ok, detail = self.satisfied(ep, history)
        return RuleResult(self.name, applicable=True, satisfied=ok, detail=detail)


RULE_REGISTRY: dict[str, type[Rule]] = {}


def register(cls: type[Rule]) -> type[Rule]:
    assert cls.name and cls.name not in RULE_REGISTRY, f"bad/duplicate rule name: {cls.name}"
    RULE_REGISTRY[cls.name] = cls
    return cls


def build_rules(names: list[str]) -> list[Rule]:
    rules = [RULE_REGISTRY[n]() for n in names]
    claimed: dict[str, str] = {}
    for r in rules:
        for f in r.field_claims:
            if f in claimed:
                raise ValueError(f"field claim conflict: {r.name} and {claimed[f]} both claim {f}")
            claimed[f] = r.name
    return rules


GREETING_RE = re.compile(r"^\s*(hi|hello|hey|dear)\b", re.IGNORECASE)


INITIALS_SUFFIX_RE = re.compile(r"\((?:[A-Za-z]\.){2,}\)\s*$")
KEBAB_RE = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")
DATE_PREFIX_RE = re.compile(r"^\d{4}-\d{2}-\d{2}:\s")


def word_count(text: str) -> int:
    return len(re.findall(r"\b[\w'-]+\b", text or ""))


def sentences(text: str) -> list[str]:
    parts = re.split(r"[.!?]+(?:\s|$)", (text or "").strip())
    return [p for p in parts if p.strip()]
