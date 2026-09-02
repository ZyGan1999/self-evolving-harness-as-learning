"""Preference rule framework.

A Rule is the unit of f*_u: (hidden semantics, programmatic checker, applicability
predicate). Rules never leak their semantics to the agent; `oracle_text` is used
only for oracle-c calibration and corrective feedback templates.

Checkers read EpisodeRecord (resolved api_calls) + SessionHistory. Field claims
prevent composing personas whose rules constrain the same text field incompatibly.
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
    satisfied: bool | None  # None iff not applicable
    detail: str = ""


class Rule:
    name: str = ""
    pool: str = ""              # "A" (in-support candidate) | "B" (out-of-support candidate)
    apps: tuple[str, ...] = ()  # apps whose presence in a task makes this rule *possibly* applicable
    trigger_actions: tuple[str, ...] = ()  # api_map ACTIONS whose invocation makes it applicable
    field_claims: tuple[str, ...] = ()  # e.g. ("gmail.body",) — conflict detection across rules
    oracle_text: str = ""       # clearest natural-language statement (for oracle-c / calibration)
    negation_text: str = ""     # assertion of the OPPOSITE preference (eta calibration, Exp-2
                                # paired candidates); empty = no natural negation
    correction_template: str = ""  # corrective-tier feedback when violated

    def trigger_apis(self) -> set[str]:
        """'{app}.{api}' strings that trigger this rule (matches ground_truth.required_apis)."""
        from .api_map import ACTIONS
        return {f"{ACTIONS[a].app}.{api}"
                for a in self.trigger_actions for api in ACTIONS[a].api_candidates}

    def gate_detail(self, ep: EpisodeRecord, history: SessionHistory) -> str:
        """Rejection text for a NON-LEAKING gate. Default: nothing beyond the template.

        `detail` exists to feed the agent the checker's computed expectation, which for some
        rules means naming the answer -- fine for the returns-value arm, fatal for a
        reject-only arm, where the cell would then measure how fast f can copy a leaked label.
        This hook is the reject-only channel: it may describe what the attempt DID (the agent's
        own action, which carries no new information) but never what it should have done.

        Subclasses that override it must keep that invariant. It is checked by
        tests/test_gate_detail_nonleaking.py, which asserts the target never appears in the
        string.
        """
        return ""

    def applicable(self, ep: EpisodeRecord) -> bool:
        raise NotImplementedError

    def satisfied(self, ep: EpisodeRecord, history: SessionHistory) -> tuple[bool, str]:
        """Only called when applicable. Returns (ok, detail).

        `detail` is not just a log line: the verifier_value arm feeds it back to the agent as
        the checker's computed expectation. Returning at the first bad action is fine when the
        expectation is per-action (a checksum over one message, a code derived from one
        amount) -- one worked example generalises. It is NOT fine when each action has a
        DIFFERENT expected value, as with a running total: there the detail must name every
        offending action, or the gate can only ever fix one of them per attempt. See
        _SpendRunningTotal.satisfied.
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


# ---- shared text helpers --------------------------------------------------

GREETING_RE = re.compile(r"^\s*(hi|hello|hey|dear)\b", re.IGNORECASE)
# Two or more dotted initials in brackets at the very end, e.g. '(J.D.)'. Checked as a FORM,
# not against the supervisor's actual initials: requiring the right letters would turn the
# preference into a profile-lookup task and stop measuring whether memory was followed.
INITIALS_SUFFIX_RE = re.compile(r"\((?:[A-Za-z]\.){2,}\)\s*$")
KEBAB_RE = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")
DATE_PREFIX_RE = re.compile(r"^\d{4}-\d{2}-\d{2}:\s")


def word_count(text: str) -> int:
    return len(re.findall(r"\b[\w'-]+\b", text or ""))


def sentences(text: str) -> list[str]:
    parts = re.split(r"[.!?]+(?:\s|$)", (text or "").strip())
    return [p for p in parts if p.strip()]
