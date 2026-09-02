"""R / D / N memory composition for the Q2 harness-size sweep.

The earlier sweep could not produce a rising branch for a structural reason: every line in
the pool described one of the d scored preferences, so raising L only ever added REDUNDANT
restatements of them. Redundancy out-votes the occasional wrong line, so the noise fraction
was bounded while the reinforcement was not, and violation could only fall.

Real memory does not grow that way. It grows mostly by accumulating correct lines about
OTHER situations, which are inapplicable to any one task. That third category -- neither
redundancy nor noise -- is what these three arms separate. All three vary only the
COMPOSITION of memory at a given L; the eval tasks, rubric and rendering are shared.

  R  redundancy   all L lines paraphrase the d scored preferences (r = 1)
  D  dilution     d lines on the scored preferences + (L-d) correct lines about preferences
                  the eval tasks never exercise (r = d/L, falling)
  N  noise        a constant fraction q of lines assert the REVERSE of a scored preference

D's filler comes from a donor persona (p7_offtopic) collected by the same learner from the
same complaint channel, so it is learner-written, not authored here. That the filler does
not collide is checked mechanically by conflict.ConflictGraph rather than asserted: every
p7 positive assertion is jointly satisfiable with every p6 positive assertion, so D moves
relevance alone. N's reverse lines come from each rule's negation_text.
"""

from __future__ import annotations

import random

from .exp2 import _render_memory, rank_within_rule
from .rules.base import RULE_REGISTRY


# Donor admission, decided lexically and uniformly rather than line by line. Hand-picking
# which collected lines are "safe filler" would be the same authored artefact the pool design
# exists to avoid, so admission is a stated rule applied to every line:
#
#   keep iff  it names the donor's own domain  AND  names nothing in the scored domain
#
# Both halves are load-bearing on this pool. The learner wrote "Text messages should use
# kebab-case for file names" -- mechanically that assertion claims file_system.name and does
# NOT conflict with any scored rule, but its text reaches into the SMS domain, so a model
# reading it may mangle a text message. That is interference, and it has to be kept out of the
# dilution arm. It also wrote "avoid making API calls on behalf of the user without explicit
# confirmation" and one "I don't have enough context to write an accurate preference": neither
# names its own domain, and the first is a brake on acting at all, which would depress TGC in
# every D cell and be read as dilution. The first half of the rule removes both.
_SCORED_VOCAB = ("phone", "sms", "text message", "texting", "message", "venmo", "payment",
                 "note", "memo", "private", "initials", "category", "sign-off", "signoff",
                 "greeting", "recipient")
_DONOR_VOCAB = ("spotify", "playlist", "song", "like_song", "liking", "file", "filename",
                "file name", "directory", "kebab", "snake_case", "overwrite", "export")


def donor_admits(text: str) -> bool:
    low = text.lower()
    return (any(t in low for t in _DONOR_VOCAB)
            and not any(t in low for t in _SCORED_VOCAB))


# A line that reports the learner's own uncertainty is not a preference -- it says so itself.
# Keeping these would put "I don't have enough context to write an accurate preference" into
# memory as filler, which measures how a model handles a visibly broken memory, a different
# question from dilution.
_NON_ASSERTION = ("enough context", "enough information", "i'm missing", "i am missing",
                  "unable to determine", "cannot determine", "unclear from")


def filter_donor(assertions: list, testable: set[str] | None = None) -> tuple[list, list]:
    """(kept, dropped). Three stated bars, applied to every line alike:

    1. lexically inside the donor domain and outside the scored one (donor_admits)
    2. induced from a rule the ConflictGraph can actually test -- an untestable rule accepts
       the whole artefact space, so its lines constrain no artefact and read as blanket
       instructions. `file_overwrite_always` is one, and the line the learner wrote from it
       ("avoid making API calls on behalf of the user without explicit confirmation") is a
       brake on acting at all: left in, it would depress TGC in every D cell.
    3. not a self-reported non-assertion
    """
    def ok(a) -> bool:
        low = a.text.lower()
        return (donor_admits(a.text)
                and (testable is None or a.rule_name in testable)
                and not any(t in low for t in _NON_ASSERTION))
    return [a for a in assertions if ok(a)], [a for a in assertions if not ok(a)]


def _by_rule(assertions: list, pick: str) -> dict[str, list]:
    out: dict[str, list] = {}
    for a in assertions:
        out.setdefault(a.rule_name, []).append(a)
    return {r: rank_within_rule(g, pick) for r, g in out.items()}


def _cycle(groups: dict[str, list], k: int) -> list:
    """Round-robin the first k lines across rules: every rule contributes its top line
    before any contributes a second, so coverage is complete as early as possible."""
    ranked = sorted(groups, key=lambda r: (-len(groups[r]), r))
    out, depth = [], 0
    while len(out) < k and depth < max((len(g) for g in groups.values()), default=0):
        for r in ranked:
            if depth < len(groups[r]) and len(out) < k:
                out.append(groups[r][depth])
        depth += 1
    return out


def _recycle(lines: list[str], k: int) -> list[str]:
    """Repeat a short list up to length k. The donor pool and the negation bank are both
    smaller than the largest L, so filler eventually repeats verbatim. That is a property of
    a finite learner, not a flaw: R's redundancy repeats too, which is what makes the arms
    comparable at equal L."""
    if not lines:
        return []
    return [lines[i % len(lines)] for i in range(k)]


def build_arm_memory(arm: str, scored: list, donor: list, d: int, L: int,
                     pick: str = "consensus", q: float = 0.3, seed: int = 0,
                     shuffle: bool = True,
                     cap_distinct: int | None = None) -> tuple[str, dict]:
    """Render L lines under one of the three compositions. Returns (memory, diagnostics).

    Diagnostics carry the mediators the analysis needs -- how many lines are on-topic, what
    share of the block they are, and where they sit -- so a rise can be attributed to
    relevance rather than to length or to position.

    cap_distinct equalises VARIETY across arms. The admitted donor pool is much smaller than
    the scored pool, so at L=150 D would recycle ~9 texts while R had ~95 -- D would then
    differ from R in two ways at once, relevance and repetitiveness, and a rise could be read
    as either. Capping every arm to the same number of distinct texts leaves relevance as the
    only difference. Pass the donor pool's distinct count.
    """
    sg = _by_rule(scored, pick)
    # Every arm opens with the same covering prefix: one top line per scored preference. So
    # coverage is complete and identical in all three from L = d upward, the arms COINCIDE at
    # L = d, and what differs is only how the remaining L-d slots are spent. Without this the
    # N arm at small L would be short of coverage AND carrying noise, and a difference there
    # could not be attributed to either.
    cover = _cycle(sg, min(d, L))
    n_bad = int(round(q * L)) if arm == "N" else 0
    n_bad = min(n_bad, max(0, L - len(cover)))
    rest = max(0, L - len(cover) - n_bad)

    if arm == "D":
        extra_on: list[str] = []
        pool = [a.text for a in _cycle(_by_rule(donor, pick), 10**6)]
        if cap_distinct:
            pool = pool[:cap_distinct]
        off_text = _recycle(pool, rest)
    else:
        # R and N both spend their non-noise remainder on further paraphrases of the scored
        # preferences, so N is R plus noise and nothing else.
        more = [a.text for a in _cycle(sg, len(cover) + rest)][len(cover):]
        if cap_distinct:                           # match D's variety, see docstring
            seen: list[str] = []
            for t in more:
                if t not in seen and len(seen) < cap_distinct:
                    seen.append(t)
            more = seen
        extra_on = _recycle(more, rest) if more else []
        neg = [RULE_REGISTRY[r]().negation_text for r in sorted(sg)
               if RULE_REGISTRY[r]().negation_text]
        off_text = _recycle(neg, n_bad)
    if arm not in ("R", "D", "N"):
        raise ValueError(f"unknown arm {arm!r}")
    on_text = [a.text for a in cover] + (extra_on if arm != "D" else [])

    lines = on_text + off_text
    # Defenders per scored rule, counted from the assertions' own rule_name rather than by
    # matching words. This is the mediator the mechanism claim rests on: the covering prefix
    # pins each scored rule at one line, so in D the defended SHARE decays monotonically as
    # off-topic lines grow, while in R it rises. A violation that tracks share and not L is
    # dilution -- and the arms can tell those apart only because L is held equal.
    src = {a.text: a.rule_name for a in scored}
    defend: dict[str, int] = {}
    for ln in lines:
        r = src.get(ln)
        if r:
            defend[r] = defend.get(r, 0) + 1

    idx = list(range(len(lines)))
    if shuffle:                                    # position is not a treatment
        # keyed on (seed, L) only, NOT on the arm: at L = d the three arms hold the same
        # lines and must therefore render byte-identical, or they would differ by order alone.
        random.Random((seed, L).__repr__()).shuffle(idx)
    lines = [lines[i] for i in idx]
    on_pos = [p for p, i in enumerate(idx) if i < len(on_text)]
    return _render_memory(lines, seed=seed), {
        "arm": arm, "L": L, "lines": len(lines),
        "on_topic": len(on_text), "off_topic": len(off_text), "noise": n_bad,
        "relevance": round(len(on_text) / max(len(lines), 1), 4),
        "distinct_on": len(set(on_text)), "distinct_off": len(set(off_text)),
        "rules_covered": len({a.rule_name for a in cover}),
        "defenders": dict(sorted(defend.items())),
        "defended_share": {r: round(c / max(len(lines), 1), 4)
                           for r, c in sorted(defend.items())},
        "mean_on_pos": round(sum(on_pos) / len(on_pos), 2) if on_pos else None,
    }
