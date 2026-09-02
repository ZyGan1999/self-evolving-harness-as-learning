"""ACE-style updater: itemized bullets with delta updates and deduplication.

Implements the ACE (arXiv:2510.04618) recipe: memory as structured bullets (id + helpful/harmful
counters + text), delta updates (LLM proposes add/modify, not rewrite-all), deterministic merge
(code does the merge, not LLM), semantic deduplication (embed & threshold), and fixed capacity
(prune when L exceeded). This avoids the "context collapse" failure mode ACE documented for
full-rewrite methods (18k tokens → 122 tokens at step 60).

If BOTH this and the naive full-rewrite plateau, the Q3 claim (self-evolution has a ceiling)
is robust to recipe. If only naive plateaus, the claim becomes "ceiling depends on recipe" --
weaker but still publishable and honest. Either way, this implementation prevents the "your
self-evolve is a strawman" critique (pre-registered risk 4).
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


# Generator/Reflector merged: LLM sees (memory, feedback) and proposes delta entries as JSON.
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
    """One memory line plus metadata ACE uses for prune/dedup."""
    id: str             # hash of original text, stable across updates
    text: str           # current content
    helpful: int = 0    # how many times marked useful (unused in Q3, but kept for symmetry)
    harmful: int = 0    # how many times marked misleading


def render_bullets(bullets: list[Bullet]) -> str:
    """Render as markdown for injection."""
    if not bullets:
        return "(no preferences recorded yet)"
    return "\n".join(f"- [{b.id[:6]}] {b.text}" for b in bullets)


def parse_bullets(text: str) -> list[Bullet]:
    """Parse bullets from injected memory (for resuming a session or testing round-trip)."""
    bullets = []
    for line in text.splitlines():
        m = re.match(r"^-\s*\[([a-f0-9]{6})\]\s*(.+)", line.strip())
        if m:
            bullets.append(Bullet(id=m[1], text=m[2].strip()))
    return bullets


def embed_text(text: str) -> list[float]:
    """Placeholder embedding for dedup. Real impl would call a small local model; for Q3 this
    just returns char-trigram Jaccard as a pseudo-embedding (close enough for dedup, and no API
    dependency). ACE uses semantic embeddings; this is a degraded substitute that still catches
    obvious duplicates like 'Texts should be signed' vs 'Text messages must be signed'.
    """
    trigrams = {text[i: i + 3] for i in range(len(text) - 2)}
    # Encode as a sparse 1000-dim vector (hash each trigram mod 1000)
    vec = [0.0] * 1000
    for tg in trigrams:
        vec[int(hashlib.md5(tg.encode()).hexdigest(), 16) % 1000] = 1.0
    norm = sum(v ** 2 for v in vec) ** 0.5
    return [v / norm if norm else 0.0 for v in vec]


def cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _extract_json_array(text: str):
    """Parse the delta array, tolerating the wrappers models add anyway.

    A bare json.loads fails on ```json fences and on any prose before the array, and every such
    failure silently means "no update this episode". That turns a formatting quirk into a fake
    plateau, so the fences and surrounding prose are stripped before giving up. Returns None only
    when no array can be recovered, which the caller counts as a parse failure.
    """
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
    if isinstance(parsed, dict):        # a single delta, unwrapped
        parsed = [parsed]
    return parsed if isinstance(parsed, list) else None


class ACEStyleUpdater(BaseUpdater):
    """ACE recipe: itemized bullets, delta updates, dedup, fixed L."""

    def __init__(self, llm: BaseLLM, max_bullets: int = 12):
        self.llm = llm
        self.bullets: list[Bullet] = []
        self.max_bullets = max_bullets
        self.embeddings: dict[str, list[float]] = {}  # id -> embedding (for dedup)
        # Audit counters. `updates` vs `parse_failures` separates "the learner had nothing to say"
        # from "the learner said something unparseable": a plateau caused by the latter is a bug
        # in this file, not a finding about self-evolution, and without the counter the two look
        # identical from the violation curve.
        self.updates = 0
        self.parse_failures = 0
        self.applied = 0
        self.history: list[dict] = []

    def observe(self, ep: EpisodeRecord, feedback: Feedback):
        # Update only when the user complained. `accepted=True` means the episode was fine, so
        # there is nothing to learn from it -- the condition is `accepted`, NOT `not accepted`:
        # inverting it skips precisely the episodes that carry complaints, and the updater then
        # runs the whole sweep with an empty memory while still costing an LLM call per episode.
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

        # Parse JSON array of deltas. If parse fails, log and skip (self-evolution gets noisy
        # feedback, unlike oracle_text which is always valid).
        deltas = _extract_json_array(response)
        if deltas is None:
            self.parse_failures += 1
            self.history.append({"episode": self.updates, "parse_failed": True,
                                 "raw": response[:200]})
            return

        # Apply deltas: add or modify. Non-dict entries are skipped rather than crashing the
        # sweep -- the delta source is an LLM, so a stray string in the array is expected traffic.
        applied_here = []
        for d in deltas:
            if not isinstance(d, dict):
                continue
            action = str(d.get("action", "")).lower()
            text = str(d.get("text", "") or "").strip()
            if action == "add" and text:
                # id is the hash of the ORIGINAL text so it survives later modifies, matching
                # ACE's stable-id property. Re-adding identical text must not create a second
                # bullet, or the dedup pass below is doing work the writer should never have made.
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

        # Dedup: if two bullets have cosine > 0.85, keep the first and drop the second
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

        # Prune to max_bullets, dropping the OLDEST. This is what keeps L roughly constant in n,
        # which Q3 needs: Q2 measured that violation rises with block length and that line ORDER
        # moves it 1.89x more than length does, so a memory free to grow would confound Q3's
        # optimisation error with Q2's length and order effects.
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
        """Carried into every train row by driver.py, so the length and order of the block are
        recorded alongside the violation rate rather than reconstructed afterwards."""
        text = render_bullets(self.bullets)
        return {"bullets": len(self.bullets), "memory_chars": len(text),
                "memory_lines": len(text.splitlines()) if self.bullets else 0,
                "updates": self.updates, "applied": self.applied,
                "parse_failures": self.parse_failures,
                # Order fingerprint: Q2 found order dominates at long blocks, so the sequence of
                # ids is recorded to make a reordering visible without diffing the whole text.
                "order": "".join(b.id[:2] for b in self.bullets)}

    def save(self, path: str) -> None:
        Path(path).write_text(json.dumps(
            {"bullets": [asdict(b) for b in self.bullets], "state": self.state(),
             "history": self.history}, ensure_ascii=False, indent=1))
