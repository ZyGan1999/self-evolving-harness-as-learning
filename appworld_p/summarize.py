"""Learner-side induction: turn an instance-tier complaint into a general preference.

This is the step that makes memory imperfect. The learner sees one complaint about one
episode and has to guess what the standing preference is -- in particular its SCOPE, which
the complaint never states. "sms not signed with first name: 'Hi Alice, the package
arrived!'" is consistent with "sign every text", "sign texts to Alice", "sign texts about
deliveries", and the learner has no way to tell from a single case. That guess is the noise
the L-sweep is about, and it is generated, not authored: nothing here reads oracle_text,
negation_text, correction_template, or even the rule's name.
"""

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from .episode import EpisodeRecord
from .feedback import Feedback
from .llm import BaseLLM
from .updaters import BaseUpdater

SUMMARIZER_SYSTEM = """You maintain a long-term memory of one user's personal preferences \
for how their assistant should do things.

The user has just complained about something the assistant did for them. From that \
complaint, write the standing preference you will store in memory, so that future tasks are \
done the way this user wants.

Rules for your output:
- Exactly one sentence, written as a preference of the user ("Text messages should ...").
- State it at the level of generality you actually believe. Do not hedge with "maybe".
- Do not mention this particular task, contact, or amount unless you believe the \
preference is genuinely specific to it.
- Output only the sentence, with no preamble, quotes, or bullet.
"""

USER_TEMPLATE = """Task the user asked for:
{instruction}

What the assistant did:
{actions}

What the user said about it:
{complaint}"""


@dataclass
class Assertion:
    """One generated memory line, plus the provenance the harness needs and the learner
    never sees."""
    rule_name: str          # which persona rule the complaint came from
    text: str               # what the learner wrote
    episode_index: int      # 1-based position in the interaction stream
    complaint: str          # the instance-tier line it was induced from


def render_actions(ep: EpisodeRecord, max_calls: int = 12) -> str:
    """The assistant's own writes, as the learner would recall them."""
    lines = []
    for call in ep.api_calls[:max_calls]:
        if call.method and call.method.lower() == "get":
            continue        # reads carry no preference signal
        args = {k: v for k, v in (call.arguments or {}).items()
                if not re.search(r"token|password|session", str(k), re.I)}
        lines.append(f"- {call.app}.{call.api}({json.dumps(args, default=str)[:400]})")
    return "\n".join(lines) or "- (no write actions)"


def induce_assertion(llm: BaseLLM, ep: EpisodeRecord, complaint: str) -> str:
    """One LLM call: complaint about this episode -> one standing preference sentence."""
    text = llm.generate(
        SUMMARIZER_SYSTEM,
        [{"role": "user", "content": USER_TEMPLATE.format(
            instruction=ep.instruction, actions=render_actions(ep), complaint=complaint)}],
        max_tokens=200, temperature=0.7)
    # Keep the first non-empty line; strip bullets/quotes the model may add anyway.
    for line in text.strip().splitlines():
        line = line.strip().lstrip("-*• ").strip().strip('"')
        if line:
            return line
    return ""


class AssertionCollector(BaseUpdater):
    """Collects the candidate pool online; memory stays empty during collection.

    Generation is online (the learner induces from each complaint as it arrives, with only
    the episodes it has seen), but the L-sweep selects from the finished pool offline. Left
    fully online, memory length would grow with experience and L would be inseparable from
    n; and build_memory()'s nesting guarantee -- raising L only ADDS lines -- would be lost.
    """

    def __init__(self, llm: BaseLLM):
        self.llm = llm
        self.assertions: list[Assertion] = []
        self.events: list[dict] = []
        self._index = 0

    def observe(self, ep: EpisodeRecord, feedback: Feedback) -> None:
        self._index += 1
        complaints = [ln.lstrip("- ").strip() for ln in feedback.text.splitlines()
                      if ln.startswith("- ")]
        written = []
        # provenance is positional: complaint i came from provenance[i] (feedback.py)
        for rule_name, complaint in zip(feedback.provenance, complaints):
            text = induce_assertion(self.llm, ep, complaint)
            if not text:
                continue
            self.assertions.append(Assertion(rule_name, text, self._index, complaint))
            written.append({"rule": rule_name, "text": text})
        self.events.append({"episode_index": self._index, "task_id": ep.task_id,
                            "accepted": feedback.accepted, "written": written})

    def render_memory(self) -> str:
        return ""           # fixed data-collection policy: empty memory while collecting

    def state(self) -> dict:
        return {"assertions": len(self.assertions), "episodes": self._index}

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(
            {"assertions": [asdict(a) for a in self.assertions], "events": self.events},
            ensure_ascii=False, indent=1))
