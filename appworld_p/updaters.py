"""Harness updaters — the "learning algorithms" over c.

All updaters see ONLY what a user-facing channel would legitimately expose:
  - default tier:    accept / generic dissatisfaction
  - corrective tier: the correction sentences (mapped back to candidates by exact
                     template match — the deterministic parse a real system would do)
  - bypass tier:     structured per-rule verdicts (UnbiasedStructured only)
They never read RuleResult ground truth directly.

Candidate bank = persona rules + distractor rules (so learning is non-trivial):
each candidate carries the assertion text that would be written into memory.
"""

import random
from dataclasses import dataclass, field

from .episode import EpisodeRecord
from .feedback import Feedback
from .llm import BaseLLM
from .rules import RULE_REGISTRY


@dataclass
class Candidate:
    name: str
    assertion: str            # text written into memory if selected
    correction: str           # correction_template (reverse-map key for corrective tier)
    alpha: float = 1.0        # Beta posterior: evidence user wants this
    beta: float = 3.0         # prior skepticism: don't write without evidence

    @property
    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)


def build_candidate_bank(persona_rule_names: list[str],
                         distractor_names: list[str] | None = None,
                         shuffle_seed: int | None = None) -> list[Candidate]:
    """Candidate hypotheses the updater selects among.

    shuffle_seed MUST be set for any arm whose n=0 checkpoint is read as a naive starting
    point. AssertionTopL sorts by posterior mean with a stable sort, so before any evidence
    arrives every mean is the prior and the order is insertion order -- and the true rules
    are listed first here. With L=6 of a bank of 8 that put both true rules in memory at
    n=0, so the "learned" arm began already holding the answer and its learning curve had
    no room to move. Shuffling deterministically by seed makes position uninformative.
    """
    names = list(persona_rule_names) + [n for n in (distractor_names or [])
                                        if n not in persona_rule_names]
    if shuffle_seed is not None:
        random.Random(shuffle_seed).shuffle(names)
    bank = []
    for n in names:
        rule = RULE_REGISTRY[n]()
        bank.append(Candidate(name=n, assertion=rule.oracle_text,
                              correction=rule.correction_template))
    return bank


class BaseUpdater:
    """observe() after each training episode; render_memory() gives current c."""

    def observe(self, ep: EpisodeRecord, feedback: Feedback) -> None:
        raise NotImplementedError

    def render_memory(self) -> str:
        raise NotImplementedError

    def state(self) -> dict:
        return {}


class NoOpUpdater(BaseUpdater):
    def observe(self, ep, feedback):
        pass

    def render_memory(self):
        return ""


class _BankedUpdater(BaseUpdater):
    def __init__(self, bank: list[Candidate]):
        self.bank = bank
        self._by_correction = {c.correction: c for c in bank}

    def observe(self, ep, feedback: Feedback):
        if feedback.tier == "corrective" and not feedback.accepted:
            for line in feedback.text.splitlines():
                cand = self._by_correction.get(line.lstrip("- ").strip())
                if cand:
                    cand.alpha += 1.0
        # default-tier reject: evidence exists but is unattributable — spread a small
        # amount over candidates whose apps were touched this episode (the q knob).
        elif feedback.tier == "default" and not feedback.accepted:
            touched_apps = {c.app for c in ep.api_calls}
            hit = [c for c in self.bank
                   if set(RULE_REGISTRY[c.name].apps) & touched_apps]
            for cand in hit:
                cand.alpha += 1.0 / max(len(hit), 1)

    def state(self):
        return {c.name: round(c.mean, 4) for c in self.bank}


class AssertionTopL(_BankedUpdater):
    """Write the L candidates with highest posterior mean as hard assertions."""

    def __init__(self, bank: list[Candidate], L: int):
        super().__init__(bank)
        self.L = L

    def render_memory(self):
        top = sorted(self.bank, key=lambda c: c.mean, reverse=True)[: self.L]
        if not top:
            return ""
        lines = ["Things I know about this user's preferences:"]
        lines += [f"- {c.assertion}" for c in top]
        return "\n".join(lines)


class ThresholdWriter(_BankedUpdater):
    """Write only candidates whose posterior mean exceeds p_bar (the Lepski-style
    restraint mechanism; eliminates the U shape per Theorem-2 prescription)."""

    def __init__(self, bank: list[Candidate], p_bar: float = 0.6, max_L: int | None = None):
        super().__init__(bank)
        self.p_bar = p_bar
        self.max_L = max_L

    def render_memory(self):
        confident = sorted((c for c in self.bank if c.mean >= self.p_bar),
                           key=lambda c: c.mean, reverse=True)
        if self.max_L is not None:
            confident = confident[: self.max_L]
        if not confident:
            return ""
        lines = ["Things I know about this user's preferences:"]
        lines += [f"- {c.assertion}" for c in confident]
        return "\n".join(lines)


class UnbiasedStructured(BaseUpdater):
    """External unbiased estimator: consumes bypass-tier structured verdicts only.
    Per-rule violation/satisfaction counts written as structured memory."""

    def __init__(self, bank: list[Candidate], min_violations: int = 1):
        self.bank = {c.name: c for c in bank}
        self.violations: dict[str, int] = {}
        self.satisfactions: dict[str, int] = {}
        self.min_violations = min_violations

    def observe(self, ep, feedback: Feedback):
        for rule_name, ok in feedback.structured.items():
            key = rule_name
            if ok:
                self.satisfactions[key] = self.satisfactions.get(key, 0) + 1
            else:
                self.violations[key] = self.violations.get(key, 0) + 1

    def render_memory(self):
        chosen = [name for name, v in self.violations.items()
                  if v >= self.min_violations and name in self.bank]
        if not chosen:
            return ""
        lines = ["User preference profile (tracked statistics):"]
        lines += [f"- {self.bank[n].assertion}" for n in sorted(chosen)]
        return "\n".join(lines)

    def state(self):
        return {"violations": dict(self.violations), "satisfactions": dict(self.satisfactions)}


SELF_EVOLVE_PROMPT = """\
You maintain a memory file about a user's preferences for their personal assistant.

Current memory:
<memory>
{memory}
</memory>

In the latest task, the assistant did the following (summary of actions):
{actions}

The user's reaction:
{feedback}

Rewrite the memory file to better capture this user's preferences. Keep it concise
(markdown bullets). Output ONLY the new memory content, nothing else.
"""


class SelfEvolveUpdater(BaseUpdater):
    """f reflects on the transcript and rewrites memory (community-standard recipe)."""

    def __init__(self, llm: BaseLLM, max_memory_chars: int = 4000):
        self.llm = llm
        self.memory = ""
        self.max_memory_chars = max_memory_chars
        self.updates = 0

    def observe(self, ep, feedback: Feedback):
        # An accepted episode carries no complaint, so there is nothing to induce from. Rewriting
        # on acceptance is worse than a wasted call: the prompt asks for a rewrite unconditionally,
        # so the model paraphrases a memory it has no new evidence about and the block drifts
        # between checkpoints for reasons unrelated to learning. That drift would land squarely on
        # top of the effect Q3 is measuring.
        if feedback.accepted or not feedback.text:
            return
        self.updates += 1
        actions = "\n".join(f"- {c.method.upper()} {c.url} {c.arguments}"[:200]
                            for c in ep.api_calls[:40]) or "(no API calls)"
        prompt = SELF_EVOLVE_PROMPT.format(memory=self.memory or "(empty)",
                                           actions=actions, feedback=feedback.text or "(none)")
        self.memory = self.llm.generate(
            system="You are a careful memory curator.",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1024, temperature=0.7,
        ).strip()[: self.max_memory_chars]

    def render_memory(self):
        return self.memory

    def state(self):
        """Length travels with every train row. Q2 measured violation rising with block length,
        and line order moving it 1.89x more than length, so for Q3 the size and ordering of the
        block have to be on record at each checkpoint rather than reconstructed later."""
        lines = [ln for ln in self.memory.splitlines() if ln.strip()]
        return {"memory_chars": len(self.memory), "memory_lines": len(lines),
                "updates": self.updates,
                "truncated": len(self.memory) >= self.max_memory_chars}
