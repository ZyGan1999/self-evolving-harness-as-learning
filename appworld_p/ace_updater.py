"""ACE-style updater: itemized bullets with delta updates and deduplication.
"""

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from .episode import EpisodeRecord
from .feedback import Feedback
from .llm import BaseLLM
from .updaters import BaseUpdater


DELTA_PROMPT = """You maintain a memory of the user's preferences as itemized bullets.

Current memory:
{memory}

Latest feedback from the user:
{feedback}

Propose changes (add or modify bullets). Output a JSON array of objects:
  [{{"action": "add", "text": "...", "reason": "..."}},
   {{"action": "modify", "id": "...", "text": "...", "reason": "..."}}]

Rules:
- "add": new bullet capturing a preference the user revealed
- "modify": rewrite bullet <id> to fix an error or generalize scope
- each "text" is one sentence, written as a user preference ("Texts should ...")
- "reason": 1-2 words on why this helps (for audit, not injected)
- output ONLY valid JSON, no preamble
"""


@dataclass
class Bullet:
    """A memory bullet with its identifier and metadata."""
    id: str
    text: str
    helpful: int = 0
    harmful: int = 0


def render_bullets(bullets: list[Bullet]) -> str:
    """Render as markdown for injection."""
    if not bullets:
        return "(no preferences recorded yet)"
    return "\n".join(f"- [{b.id[:6]}] {b.text}" for b in bullets)


def parse_bullets(text: str) -> list[Bullet]:
    """Parse bullet identifiers and text from rendered memory."""
    bullets = []
    for line in text.splitlines():
        m = re.match(r"^-\s*\[([a-f0-9]{6})\]\s*(.+)", line.strip())
        if m:
            bullets.append(Bullet(id=m[1], text=m[2].strip()))
    return bullets


def embed_text(text: str) -> list[float]:
    """Encode text as character trigrams for local deduplication."""
    trigrams = {text[i: i + 3] for i in range(len(text) - 2)}

    vec = [0.0] * 1000
    for tg in trigrams:
        vec[int(hashlib.md5(tg.encode()).hexdigest(), 16) % 1000] = 1.0
    norm = sum(v ** 2 for v in vec) ** 0.5
    return [v / norm if norm else 0.0 for v in vec]


def cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _extract_json_array(text: str):
    """Parse a JSON delta array, allowing code fences and surrounding prose."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("["), text.rfind("]")
        if start == -1 or end <= start:
            return None
        try:
            parsed = json.loads(text[start: end + 1])
        except json.JSONDecodeError:
            return None
    if isinstance(parsed, dict):
        parsed = [parsed]
    return parsed if isinstance(parsed, list) else None


class ACEStyleUpdater(BaseUpdater):
    """ACE-style updater with delta updates, deduplication, and bounded memory."""

    def __init__(self, llm: BaseLLM, max_bullets: int = 12):
        self.llm = llm
        self.bullets: list[Bullet] = []
        self.max_bullets = max_bullets
        self.embeddings: dict[str, list[float]] = {}

        self.updates = 0
        self.parse_failures = 0
        self.applied = 0
        self.history: list[dict] = []

    def observe(self, ep: EpisodeRecord, feedback: Feedback):
        if feedback.accepted or not feedback.text:
            return
        self.updates += 1

        prompt = DELTA_PROMPT.format(memory=render_bullets(self.bullets),
                                     feedback=feedback.text)
        response = self.llm.generate(
            system="You are a memory curator.",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=512, temperature=0.7,
        ).strip()

        deltas = _extract_json_array(response)
        if deltas is None:
            self.parse_failures += 1
            self.history.append({"episode": self.updates, "parse_failed": True,
                                 "raw": response[:200]})
            return

        applied_here = []
        for d in deltas:
            if not isinstance(d, dict):
                continue
            action = str(d.get("action", "")).lower()
            text = str(d.get("text", "") or "").strip()
            if action == "add" and text:
                new_id = hashlib.md5(text.encode()).hexdigest()[:6]
                if any(b.id == new_id for b in self.bullets):
                    continue
                self.bullets.append(Bullet(id=new_id, text=text))
                self.embeddings[new_id] = embed_text(text)
                applied_here.append({"action": "add", "id": new_id, "text": text})
            elif action == "modify" and text:
                target = str(d.get("id", "") or "").strip().lstrip("[").rstrip("]")[:6]
                for b in self.bullets:
                    if target and b.id.startswith(target):
                        b.text = text
                        self.embeddings[b.id] = embed_text(text)
                        applied_here.append({"action": "modify", "id": b.id, "text": text})
                        break

        seen = []
        deduped = []
        for b in self.bullets:
            emb = self.embeddings.get(b.id)
            if emb and any(cosine(emb, e) > 0.85 for e in seen):
                continue
            deduped.append(b)
            if emb:
                seen.append(emb)
        dropped_dup = len(self.bullets) - len(deduped)
        self.bullets = deduped

        dropped_cap = 0
        if len(self.bullets) > self.max_bullets:
            dropped = self.bullets[: len(self.bullets) - self.max_bullets]
            self.bullets = self.bullets[-self.max_bullets:]
            dropped_cap = len(dropped)
            for b in dropped:
                self.embeddings.pop(b.id, None)

        self.applied += len(applied_here)
        self.history.append({"episode": self.updates, "deltas": len(deltas),
                             "applied": applied_here, "dropped_dup": dropped_dup,
                             "dropped_cap": dropped_cap, "bullets_after": len(self.bullets)})

    def render_memory(self) -> str:
        return render_bullets(self.bullets)

    def state(self) -> dict:
        """Return memory size, update diagnostics, and a bullet-order fingerprint."""
        text = render_bullets(self.bullets)
        return {"bullets": len(self.bullets), "memory_chars": len(text),
                "memory_lines": len(text.splitlines()) if self.bullets else 0,
                "updates": self.updates, "applied": self.applied,
                "parse_failures": self.parse_failures,

                "order": "".join(b.id[:2] for b in self.bullets)}

    def save(self, path: str) -> None:
        Path(path).write_text(json.dumps(
            {"bullets": [asdict(b) for b in self.bullets], "state": self.state(),
             "history": self.history}, ensure_ascii=False, indent=1))
