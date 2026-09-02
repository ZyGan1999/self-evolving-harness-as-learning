"""Mechanical conflict detection between directed preference assertions.

The Q2 rising branch needs distractor assertions that CONTRADICT the persona. Which ones
those are must not be a judgement call: a hand-maintained INTERFERING list is exactly the
artefact a reviewer reads as "the authors labelled which distractors would hurt".

So conflict is decided by executing the rules' own checkers. Two directed assertions
conflict iff they are JOINTLY UNSATISFIABLE -- no artefact in a candidate space passes
both. The candidate space is built from the same ACTIONS table the checkers match against
(itself derived from AppWorld's API docs), so nothing here knows what a persona is.

Directions matter and rules alone are not the unit. `payment_has_note` asserted positively
("always leave a note") is compatible with "notes start with [personal]"; asserted in
REVERSE ("never leave a note") it is not. Memory writes whichever direction the posterior
favours, so the conflict graph is over (rule, direction) pairs. A rule's negative
accepting set is the complement of its positive one within the candidate space.

Coverage is the one real limitation. A rule whose accepting set is the whole space, or
empty, carries no information here and is dropped as untestable -- omitting the empty
string from the space once silently dropped `payment_has_note`, whose positive form then
looked vacuously true. `untestable()` reports these so a caller can see what was skipped
rather than reading silence as absence of conflict.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

from .apicalls import ApiCall
from .episode import EpisodeRecord
from .rules.api_map import ACTIONS

# Checkers read a text field under whichever name their API uses; `ApiCall.arg` tries
# candidates in order. Setting every plausible name to the same string keeps the space
# action-generic instead of hard-coding one argument name per app.
TEXT_ARGS = ("description", "note", "memo", "message", "content", "body", "title",
             "subject", "name", "file_path")

# In-support rules as authored at M0 (4e458fd, 2026-07-07) -- the commit that first built
# the rule bank, three weeks before Q2 existed. Cross-borrowing draws BOTH persona and
# distractors from here, so no rule in the experiment was written with knowledge of which
# persona it would sit beside.
#
# `private_note_format` is included: authored at M0 as pool B, relabelled A at 099dcf9
# (2026-07-27) because calibration measured oracle 1.00 against a 0.00 baseline, i.e. a
# perfect description sufficed. That relabel predates Q2 by a week and was driven by a
# measurement, not by conflict structure -- it happens to claim `venmo.note`, the same field
# as `payment_has_note`, which is why the pool has any same-field pair at all.
#
# Everything added on 2026-08-02 (a5420a4) and 2026-08-03 (b623c00, 8dc10b9) is EXCLUDED:
# those are the note-subfield rules and the seven hand-picked interfering distractors, all
# authored after the rising branch was the open question. Their presence is precisely what a
# reviewer would read as the answer being built into the pool.
M0_POOL_A = (
    "email_greeting_signoff", "email_subject_short", "file_kebab_case",
    "note_tags_required", "note_title_dated", "payment_has_note",
    "playlist_over_like", "playlist_private", "private_note_format",
    "sms_signoff", "todoist_due_date", "todoist_title_capitalized", "venmo_private",
)

_SUPERVISOR = {"first_name": "John", "last_name": "Doe", "email": "john.doe@example.com",
               "phone_number": "555-0100"}

# Morphological features spanning what the M0 checkers inspect: presence, prefix tag,
# greeting, sign-off, initials, bracketing, word count, case, date stamp, tags. The product
# is deliberately crossed rather than curated -- a feature combination nobody thought about
# is the point, since it is what makes joint satisfiability a search and not an assertion.
_PREFIXES = ("", "[personal] ", "Hi Alice, ", "hey ", "2026-08-04: ")
_BODIES = ("", "coffee", "coffee and bagels for the weekend trip")
_SUFFIXES = ("", " (J.D.)", " -John", "!", " #personal #trip")
_CASES = (lambda s: s, str.lower, str.upper, str.title)

# Structured arguments are toggled, not fixed. Pinning `tags` and `due_date` to a present
# value made `note_tags_required` and `todoist_due_date` accept every artefact, and the
# absent case is the one their negations assert -- a rule that cannot be violated in the
# space is a rule the graph silently drops.
_STRUCTURED = (
    {"tags": ["personal"], "due_date": "2026-08-10", "is_public": True, "private": True},
    {"tags": ["personal"], "due_date": "2026-08-10", "is_public": False, "private": False},
    {"tags": [], "due_date": None, "is_public": True, "private": False},
    {"tags": [], "due_date": None, "is_public": False, "private": True},
)


def candidate_texts() -> tuple[str, ...]:
    """Crossed morphological space, plus degenerate long/short cases."""
    out = {case(p + b + s)
           for p, b, s in itertools.product(_PREFIXES, _BODIES, _SUFFIXES)
           for case in _CASES}
    out.add("x")                      # single short word, no structure
    out.add("kebab-case-title")       # satisfies KEBAB_RE, which the crossed space misses
    out.add("word " * 60)             # far past any length threshold
    return tuple(sorted(out))


def _omission_patterns(actions: tuple[str, ...]) -> list[frozenset[str]]:
    """Action sets to build episodes from: all present, then each one omitted in turn.

    Some rules are about what the agent did INSTEAD of something -- `playlist_over_like`
    passes only when no like_song call is present. With every action always present such a
    rule rejects the whole space and drops out, so absence has to be reachable.
    """
    full = frozenset(actions)
    return [full] + [full - {a} for a in actions]


def _call_for(action: str, text: str, structured: dict) -> ApiCall:
    matcher = ACTIONS[action]
    args: dict = {name: text for name in TEXT_ARGS}
    args["file_path"] = text or "notes.txt"           # must stay path-shaped
    args.update(amount=10.0, receiver_email="alice@example.com",
                phone_number="555-0199", email_addresses=["alice@example.com"],
                overwrite=True, song_id="s1", playlist_id="p1", project_id="pr1",
                status="completed", answer="done")
    args.update({k: v for k, v in structured.items() if v is not None})
    args["is_private"] = structured["private"]
    return ApiCall(app=matcher.app, api=matcher.api_candidates[0], arguments=args,
                   method=matcher.method)


def iter_artifacts(actions: tuple[str, ...]):
    """Deterministic stream of candidate artefacts spanning `actions`.

    Every artefact carries one call per included action with a shared text and shared
    structured arguments, so all rules over these actions are applicable to the same
    artefacts and their accepting sets are directly comparable. Yielded lazily and in fixed
    order: accepting sets are recorded as indices into this stream, so it is never stored.
    """
    for pattern in _omission_patterns(actions):
        for text in candidate_texts():
            for structured in _STRUCTURED:
                calls = [_call_for(a, text, structured) for a in actions if a in pattern]
                if not calls:
                    continue
                yield EpisodeRecord(
                    task_id="conflict-probe", instruction="probe",
                    supervisor=dict(_SUPERVISOR), api_calls=calls, task_completed=True)


@dataclass(frozen=True)
class Assertion:
    """A memory line: a rule asserted in one direction. '+' writes pos_text, '-' neg_text."""
    rule: str
    direction: str

    def __str__(self) -> str:
        return f"{self.rule}[{self.direction}]"


def _accepts(rule, episode) -> bool | None:
    """Does the checker pass this artefact? None = inapplicable or checker error."""
    try:
        if not rule.applicable(episode):
            return None
        satisfied, _ = rule.satisfied(episode, None)
        return bool(satisfied)
    except Exception:
        # A checker that needs SessionHistory (pool-B aggregates) cannot be probed from a
        # single synthetic episode. Reported by untestable(), not silently treated as pass.
        return None


class ConflictGraph:
    """Joint-satisfiability conflict graph over directed assertions.

    Built by running each rule's checker over the candidate space for the action it
    triggers, then intersecting accepting sets. No rule identity, persona membership, or
    harm label enters the computation.
    """

    def __init__(self, rules: dict):
        self._action: dict[str, str] = {}
        self._accept: dict[Assertion, frozenset[int]] = {}
        self._untestable: dict[str, str] = {}
        self._build(rules)

    def _build(self, rules: dict) -> None:
        instances = {}
        for name, rule in sorted(rules.items()):
            instance = rule() if isinstance(rule, type) else rule
            if not instance.trigger_actions:
                self._untestable[name] = "no trigger_actions"
                continue
            instances[name] = instance
        # One shared artefact space over every action the pool touches, so that any two
        # rules are compared on the same artefacts. Restricting each rule to its own action
        # would make accepting sets incomparable and every cross-app pair vacuously safe.
        self._actions = tuple(sorted({a for i in instances.values()
                                      for a in i.trigger_actions}))
        pos: dict[str, set[int]] = {n: set() for n in instances}
        neg: dict[str, set[int]] = {n: set() for n in instances}
        for index, episode in enumerate(iter_artifacts(self._actions)):
            for name, instance in instances.items():
                verdict = _accepts(instance, episode)
                if verdict is True:
                    pos[name].add(index)
                elif verdict is False:
                    neg[name].add(index)
        self._size = index + 1
        for name in instances:
            positive, negative = frozenset(pos[name]), frozenset(neg[name])
            if not positive and not negative:
                self._untestable[name] = "checker never applied in the candidate space"
                continue
            if not negative:
                self._untestable[name] = "accepts the whole space (no discriminating case)"
                continue
            if not positive:
                self._untestable[name] = "rejects the whole space (no satisfying case)"
                continue
            self._action[name] = instances[name].trigger_actions
            # '-' asserts the rule must be VIOLATED, so its satisfying artefacts are the
            # ones the checker rejects -- not merely the complement, since inapplicable
            # artefacts belong to neither side.
            self._accept[Assertion(name, "+")] = positive
            self._accept[Assertion(name, "-")] = negative

    def testable(self) -> list[str]:
        return sorted(self._action)

    def untestable(self) -> dict[str, str]:
        """Rules excluded from the graph, with the reason. Coverage gaps live here."""
        return dict(self._untestable)

    def conflicts(self, a: Assertion, b: Assertion) -> bool:
        """True iff no candidate artefact satisfies both assertions.

        Assertions on different actions never conflict: they constrain disjoint artefacts,
        so an agent can satisfy both in one episode.
        """
        if a.rule == b.rule:
            return a.direction != b.direction      # a rule contradicts its own negation
        if self._action.get(a.rule) != self._action.get(b.rule):
            return False
        left, right = self._accept.get(a), self._accept.get(b)
        if not left or not right:
            return False
        return not (left & right)

    def against(self, assertion: Assertion, targets: list[Assertion]) -> list[Assertion]:
        """Which of `targets` this assertion is jointly unsatisfiable with."""
        return [t for t in targets if self.conflicts(assertion, t)]

    def conflict_count(self, written: list[Assertion], persona: list[Assertion]) -> int:
        """Collisions in a realized memory: written non-persona lines contradicting persona.

        This is the Q2 dose variable. It is measured from a memory the learner produced,
        never chosen -- which distractors land in the top-L is the posterior's business.
        """
        persona_rules = {p.rule for p in persona}
        return sum(1 for w in written
                   if w.rule not in persona_rules and self.against(w, persona))

    def collided_persona_rules(self, written: list[Assertion],
                               persona: list[Assertion]) -> dict[str, list[str]]:
        """Persona rule -> the written assertions contradicting it.

        Aggregate violation rate over the whole persona dilutes the dose: in the pilot two
        memories with three collisions each scored 0.500 and 0.000, and the difference was
        WHICH persona rule got hit -- one collision landed on the note text the persona
        also constrains, the other on an unrelated boolean flag. Scoring the violation of
        the specific rule a collision targets is what makes the dose-response meaningful.
        """
        persona_rules = {p.rule for p in persona}
        hits: dict[str, list[str]] = {p.rule: [] for p in persona}
        for w in written:
            if w.rule in persona_rules:
                continue
            for target in self.against(w, persona):
                hits[target.rule].append(str(w))
        return hits
