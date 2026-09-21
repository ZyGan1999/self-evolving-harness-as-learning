"""Q1 candidate-bank selection and Q3 full-memory rewriting."""

import random
from dataclasses import dataclass

from .episode import EpisodeRecord
from .feedback import Feedback
from .llm import BaseLLM
from .rules import RULE_REGISTRY


@dataclass
class Candidate:
    name: str
    assertion: str
    correction: str
    alpha: float = 1.0
    beta: float = 3.0

    @property
    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)


def build_candidate_bank(persona_rule_names: list[str],
                         distractor_names: list[str] | None = None,
                         shuffle_seed: int | None = None) -> list[Candidate]:
    """Build the candidate bank in a seeded order for deterministic tie-breaking."""
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
        """Return memory size and update counts."""
        lines = [ln for ln in self.memory.splitlines() if ln.strip()]
        return {"memory_chars": len(self.memory), "memory_lines": len(lines),
                "updates": self.updates,
                "truncated": len(self.memory) >= self.max_memory_chars}
