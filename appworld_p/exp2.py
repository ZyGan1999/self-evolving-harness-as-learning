"""Exp 2 (Q2, estimation error) machinery: paired candidates + offline grid reconstruction.

Design (see refine-logs/EXPERIMENT_PLAN.md Block 2):
- Each candidate *attribute* is binary: the user wants P_r (positive) or not-P_r
  (negative). Persona attributes have true direction = positive; distractor
  attributes have no true preference.
- The interaction stream is collected ONCE under a fixed data-collection policy
  (empty memory). Every episode logs user-visible signals only:
    * behavior verdicts per candidate rule (checkers are behavior predicates,
      independent of the hidden persona),
    * the acceptance bit, and the corrective sentences (mappable to rules).
- Posteriors for any (n, feedback-tier) are replayed offline from the event log;
  memory for any (L, variant) is constructed from the posterior. Only evaluation
  costs LLM calls -> the L x n grid needs a single collection run.

Evidence model (noisy binary observation of each attribute):
  corrective tier: correction naming rule r        -> +1.0 toward positive
  default tier   : episode rejected & behavior lacked P_r -> +1/k toward positive
                   episode rejected & behavior had  P_r   -> +1/k toward negative
  both tiers     : episode accepted & behavior had  P_r   -> +w toward positive
                   episode accepted & behavior lacked P_r -> +w toward negative
  (w = ACCEPT_WEIGHT, k = number of applicable candidates that episode)
"""

import json
import random
import re
from dataclasses import dataclass, field
from pathlib import Path

from .episode import EpisodeRecord
from .feedback import Feedback
from .history import SessionHistory
from .rules import RULE_REGISTRY
from .updaters import BaseUpdater

ACCEPT_WEIGHT = 0.25

# Innocuous filler for the token-padding control (rising branch must survive
# equal-length contexts, otherwise it is attention dilution, not statistics).
FILLER_SENTENCE = ("Background note: the user lives in a temperate city, commutes on "
                   "weekdays, and generally checks messages in the evening. ")

# Padding has to be VARIED, not one sentence repeated. Filling 32k characters with 230 copies
# of the same clause is a degenerate stimulus: exact repetition is trivially compressible and a
# model can learn to skip it, which would understate dilution and make a flat curve
# uninformative. These fragments combine into distinct, innocuous sentences that carry no
# preference and never mention a constrained field.
_FILLER_SUBJECTS = ("the user", "this account holder", "the household", "the primary user")
_FILLER_TOPICS = (
    "prefers window seats when travelling", "keeps a spare umbrella by the door",
    "waters the plants on Sunday mornings", "reads the news before breakfast",
    "walks to the corner shop for milk", "listens to the radio while cooking",
    "keeps receipts in a drawer for a month", "returns library books on time",
    "takes the stairs rather than the lift", "buys coffee beans in bulk",
    "labels leftovers with the date", "checks the weather the night before",
    "sorts recycling on Tuesday evenings", "keeps a spare key with a neighbour",
    "prefers tap water to bottled", "rotates winter and summer clothing seasonally",
)


def filler_text(n_chars: int, seed: int = 0) -> str:
    """Innocuous varied prose of at least n_chars, deterministic in seed."""
    rng = random.Random((seed, "filler", n_chars).__repr__())
    out: list[str] = []
    total = 0
    i = 0
    while total < n_chars:
        i += 1
        line = (f"Note {i}: {rng.choice(_FILLER_SUBJECTS)} "
                f"{rng.choice(_FILLER_TOPICS)}.")
        out.append(line)
        total += len(line) + 1
    return "\n".join(out)


_STOPWORDS = frozenset("""the a an should be is are to of with my your our their its and or at as
by that this it user users me you always must never when unless any all in on for from""".split())


def _content_words(text: str) -> frozenset:
    return frozenset(w for w in re.findall(r"[a-z0-9']+", text.lower())
                     if w not in _STOPWORDS)


def rank_within_rule(group: list, pick: str) -> list:
    """Order one rule's assertions. Uses only learner-visible signals -- never oracle_text,
    so this cannot smuggle in knowledge of which paraphrase is actually right.

    earliest   induction order, i.e. a memory that never revisits what it wrote
    consensus  most central paraphrase first, by mean word overlap with its own group:
               what a memory that consolidates duplicates would keep

    `earliest` is not neutral, which is why `consensus` exists. Under earliest the L=5 memory
    for this pool opened with "Text messages should not include untagged notes" and "Text
    messages should include a note with the user's initials as a suffix" -- both induced from
    venmo complaints and both mis-scoped onto SMS, because they happened to come first. So
    the L=d point, which is supposed to be exactly-enough correct coverage, was carrying two
    actively wrong lines and the descending branch had no reason to bottom out there.
    Consensus demotes them: the other paraphrases in each group talk about payments, so an
    SMS-scoped outlier is far from its own centroid. It also demotes the learner's occasional
    meta-commentary ("I will store this preference in memory for future tasks:"), which
    shares almost no content words with any real preference -- a filter that falls out of
    consolidation rather than one I applied by hand.
    """
    if pick == "earliest":
        return sorted(group, key=lambda a: a.episode_index)
    # Consolidation collapses verbatim repeats: the learner induced the same sentence from
    # several episodes, and storing it twice is the one kind of redundancy that provably
    # carries no new claim. Left in, L=8 spent two of its three extra slots on literal
    # copies of the L=1 and L=2 lines, so the rising branch would have been measuring
    # memory length -- which oracle_padded already isolates -- instead of disagreement
    # between paraphrases. `earliest` keeps them, being the non-consolidating harness.
    seen: set[str] = set()
    deduped = []
    for a in sorted(group, key=lambda a: a.episode_index):
        key = a.text.strip().lower()
        if key not in seen:
            seen.add(key)
            deduped.append(a)
    group = deduped
    words = [_content_words(a.text) for a in group]

    def centrality(i: int) -> float:
        others = [w for j, w in enumerate(words) if j != i]
        if not others:
            return 1.0
        return sum(len(words[i] & o) / len(words[i] | o) if (words[i] | o) else 1.0
                   for o in others) / len(others)

    # episode_index breaks centrality ties, so the order is total and reproducible.
    scored = [(-centrality(i), group[i].episode_index, i) for i in range(len(group))]
    return [group[i] for _, _, i in sorted(scored)]


def select_assertions(assertions: list, L: int, order: str = "roundrobin",
                      pick: str = "consensus") -> list:
    """Pick L of the pool's generated assertions.

    roundrobin  every preference gets its top-ranked assertion before any gets a second, so
                L < d is a coverage deficit and L > d is redundancy -- one mechanism per
                branch of the U instead of both at once.
    confidence  strict most-complained-first, no round-robin. A robustness control: here a
                small L can spend every slot on one preference, so if the descending branch
                is really about coverage it should be shallower under this order.

    Rules are ranked by how often the user complained about them (the only confidence
    signal the learner has), and within a rule the earliest assertion comes first. Both
    orders are prefix-nested in L: raising L only ever appends.
    """
    by_rule: dict[str, list] = {}
    for a in assertions:
        by_rule.setdefault(a.rule_name, []).append(a)
    for rule, group in by_rule.items():
        by_rule[rule] = rank_within_rule(group, pick)
    ranked = sorted(by_rule, key=lambda r: (-len(by_rule[r]), r))
    if order == "confidence":
        return [a for r in ranked for a in by_rule[r]][:L]
    if order == "covered":
        # Every preference's top line is always present, so coverage is complete and constant
        # for all L >= d and L is the only thing varying. Under roundrobin the identity of the
        # first d lines shifts with L, which conflates "more memory" with "better coverage";
        # here it cannot. L < d is clamped to d -- this order says nothing about the
        # coverage-deficit branch and is not meant to.
        base = [by_rule[r][0] for r in ranked]
        rest = [a for r in ranked for a in by_rule[r][1:]]
        return base + rest[:max(0, L - len(base))]
    out, depth = [], 0
    while len(out) < L and depth < max((len(g) for g in by_rule.values()), default=0):
        for r in ranked:
            if depth < len(by_rule[r]) and len(out) < L:
                out.append(by_rule[r][depth])
        depth += 1
    return out


def build_pool_memory(assertions: list, L: int, order: str = "roundrobin",
                      pick: str = "consensus", seed: int = 0, pad_to_chars: int = 0,
                      interleave: bool = False) -> str:
    """Memory block from L generated assertions (no oracle text anywhere)."""
    chosen = select_assertions(assertions, L, order, pick)
    return _render_memory([a.text for a in chosen], seed=seed,
                          pad_to_chars=pad_to_chars, interleave=interleave)


@dataclass
class Attribute:
    rule_name: str
    pos_text: str
    neg_text: str
    is_persona: bool
    # evidence toward positive / negative direction
    pos: float = 0.0
    neg: float = 0.0

    @property
    def p_positive(self) -> float:
        """Posterior mean of Beta(1+pos, 1+neg)."""
        return (1 + self.pos) / (2 + self.pos + self.neg)

    @property
    def confidence(self) -> float:
        return abs(self.p_positive - 0.5)


def build_attributes(persona_rule_names: list[str], distractor_names: list[str]) -> list[Attribute]:
    attrs = []
    for name in list(persona_rule_names) + [d for d in distractor_names
                                            if d not in persona_rule_names]:
        rule = RULE_REGISTRY[name]()
        if not rule.negation_text:
            continue  # paired design requires a negation
        attrs.append(Attribute(rule_name=name, pos_text=rule.oracle_text,
                               neg_text=rule.negation_text,
                               is_persona=name in persona_rule_names))
    return attrs


class EventLogger(BaseUpdater):
    """Updater used during collection: memory stays empty, events get recorded."""

    def __init__(self, bank_rule_names: list[str], correction_templates: dict[str, str]):
        self.bank_rule_names = bank_rule_names
        self._by_correction = {v: k for k, v in correction_templates.items()}
        self.events: list[dict] = []
        self._rules = {n: RULE_REGISTRY[n]() for n in bank_rule_names}
        self._history = SessionHistory()  # behavior-verdict evaluation needs a history

    def observe(self, ep: EpisodeRecord, feedback: Feedback) -> None:
        verdicts = {}
        for name, rule in self._rules.items():
            res = rule.check(ep, self._history)
            if res.applicable:
                verdicts[name] = bool(res.satisfied)
        self._history.update(ep)
        corrected = [self._by_correction[line.lstrip("- ").strip()]
                     for line in feedback.text.splitlines()
                     if line.lstrip("- ").strip() in self._by_correction]
        self.events.append({"accepted": feedback.accepted, "verdicts": verdicts,
                            "corrected_rules": corrected, "task_id": ep.task_id})

    def render_memory(self) -> str:
        return ""  # fixed data-collection policy

    def state(self) -> dict:
        return {"events": len(self.events)}

    def save(self, path: str | Path) -> None:
        Path(path).write_text("\n".join(json.dumps(e) for e in self.events))


def replay_posterior(events: list[dict], attributes: list[Attribute],
                     n: int, tier: str) -> list[Attribute]:
    """Fresh copies of attributes with evidence from the first n events under
    the given feedback tier ('corrective' = attributed, 'default' = ambiguous)."""
    attrs = {a.rule_name: Attribute(a.rule_name, a.pos_text, a.neg_text, a.is_persona)
             for a in attributes}
    for event in events[:n]:
        applicable = {r: v for r, v in event["verdicts"].items() if r in attrs}
        if event["accepted"]:
            for r, ok in applicable.items():
                if ok:
                    attrs[r].pos += ACCEPT_WEIGHT
                else:
                    attrs[r].neg += ACCEPT_WEIGHT
        elif tier == "corrective":
            for r in event["corrected_rules"]:
                if r in attrs:
                    attrs[r].pos += 1.0
        else:  # default tier: unattributed rejection
            k = max(len(applicable), 1)
            for r, ok in applicable.items():
                if ok:
                    attrs[r].neg += 1.0 / k
                else:
                    attrs[r].pos += 1.0 / k
    return list(attrs.values())


def build_memory(attributes: list[Attribute], L: int, variant: str = "learned",
                 seed: int = 0, pad_to_chars: int = 0, interleave: bool = False) -> str:
    """Construct the assertion memory for a grid point.

    learned   top-L attributes by confidence, MAP direction (ties -> seeded coin)
    oracle    top-L all-correct assertions (persona attributes only, positive text)
    threshold interpret L as max count but only write confidence >= 0.15

    interleave spreads the assertions THROUGH the padding instead of listing them first.
    Front-loaded padding measured 0.000 violations at every length up to 8k tokens, which is a
    weak test: the assertions sat at the very top of the memory block, the position a model
    reads most reliably, and the filler was all behind them. It is also not what a bloated
    memory looks like -- real ones interleave what matters with what does not, and the
    retrieval literature finds position dominates count.
    """
    # Tie-breaks are seeded PER ATTRIBUTE, never with L. Seeding with L made the sweep
    # non-nested: an attribute with no evidence at all (posterior exactly 0.5, which at n=4
    # happens to 3 of the 5 persona attributes) got a fresh coin at every L, so the same
    # posterior produced different memories along the axis. The n=4 curve that came out --
    # 0.75, 0.625, 0.625, 0.125, 0.625, 0.25 -- was those coins, not thin evidence: at L=8 two
    # of the three tied attributes landed correct, at L=11 all three landed wrong, at L=15 two
    # landed correct again. Per-attribute seeding fixes each direction given (seed, posterior),
    # so raising L only ADDS assertions and the curve is about L.
    lines: list[str] = []
    if variant == "oracle":
        chosen = [a for a in attributes if a.is_persona][:L]
        lines = [a.pos_text for a in chosen]
    else:
        ranked = sorted(attributes, key=lambda a: a.confidence, reverse=True)
        if variant == "threshold":
            ranked = [a for a in ranked if a.confidence >= 0.15]
        for a in ranked[:L]:
            if a.p_positive > 0.5:
                lines.append(a.pos_text)
            elif a.p_positive < 0.5:
                lines.append(a.neg_text)
            else:
                # no evidence either way: a fixed coin per (seed, attribute)
                coin = random.Random((seed, variant, a.rule_name).__repr__())
                lines.append(a.pos_text if coin.random() < 0.5 else a.neg_text)
    return _render_memory(lines, seed=seed, pad_to_chars=pad_to_chars,
                          interleave=interleave)


def _render_memory(lines: list[str], seed: int = 0, pad_to_chars: int = 0,
                   interleave: bool = False) -> str:
    """Shared block layout + token padding for both the paired-attribute and the
    generated-pool memories, so the two differ only in where the sentences came from."""
    if not lines:
        memory = ""
    else:
        memory = "Things I know about this user's preferences:\n" + \
                 "\n".join(f"- {t}" for t in lines)
    if pad_to_chars and interleave and lines:
        # one filler block between consecutive assertions, so the assertions are spread evenly
        # from the top of the block to the bottom rather than clustered at the top
        per = max((pad_to_chars - len(memory)) // (len(lines) + 1), 0)
        chunks = [filler_text(per, seed=(seed, i).__hash__()) for i in range(len(lines) + 1)]
        body = []
        for i, line in enumerate(lines):
            body.append(chunks[i])
            body.append(f"- {line}")
        body.append(chunks[-1])
        memory = ("Things I know about this user's preferences:\n"
                  + "\n".join(body))[:pad_to_chars]
    elif pad_to_chars and len(memory) < pad_to_chars:
        head = "\n\nGeneral notes:\n"
        want = pad_to_chars - len(memory) - len(head)
        # Truncated to EXACTLY the budget, mid-sentence if need be: equal length across L is
        # the whole point of the padding control, and letting whole-line filler overshoot by a
        # few dozen characters would reintroduce the length difference it exists to remove.
        memory = (memory + head + filler_text(max(want, 0), seed=seed))[:pad_to_chars]
    return memory
