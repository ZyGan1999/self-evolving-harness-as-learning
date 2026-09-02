"""Published self-evolution recipes, as comparable updaters for Q3.

ACE and the naive full-rewrite control occupy only two cells of the design space the literature
actually spans. The two axes that matter for a plateau claim are **capacity control** (does the
memory have a bound?) and **update granularity** (is an update a targeted edit, an append, or a
regeneration of everything?). Filling the empty cells is what stops "self-evolution plateaus"
from being a statement about one recipe:

    capacity \\ granularity   delta/targeted        append-only        regenerate-all
    fixed bound              ACE (max_bullets=12)   Reflexion (W=3)    --
    unbounded                A-MEM                  --                 Dynamic Cheatsheet,
                                                                       naive full rewrite

Admission rule: an arm must run under the supervision this experiment actually has -- a frozen
model, ONE rollout per episode, a free-form natural-language complaint, no scalar reward, no
ground-truth labels. That disqualifies several better-known methods rather than any judgement
about their quality: GEPA (arXiv:2507.19457) selects over a Pareto frontier of SCORED candidates,
ReasoningBank/MaTTS (arXiv:2509.25140) needs parallel rollouts for its contrastive signal, ExpeL
(arXiv:2308.10144) consumes same-task success/failure PAIRS, and TextGrad/OPRO/APE all optimise
against a metric. They belong in related work, not in this table.

Implemented here:
  ReflexionUpdater            Shinn et al., arXiv:2303.11366 (NeurIPS 2023)
  DynamicCheatsheetUpdater    Suzgun et al., arXiv:2504.07952 (EACL 2025)
  AMemUpdater                 Xu et al., arXiv:2502.12110 (NeurIPS 2025)

Each deviation from its paper is marked DEVIATION in the class docstring, because a baseline that
quietly differs from its citation is worse than no baseline.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .ace_updater import _extract_json_array, cosine, embed_text
from .episode import EpisodeRecord
from .feedback import Feedback
from .llm import BaseLLM
from .updaters import BaseUpdater


# Argument names whose VALUE is credential plumbing rather than anything a preference constrains.
# Redacted before the trajectory reaches any curator LLM. This is not cosmetic: with the raw
# trajectory, 6 of 18 TEPA updates on seed 0 were REFUSED by the curator, which read
# show_account_passwords + login + bearer tokens as credential theft and replied "I can't help
# with this request... unauthorized access to accounts, credential theft, and fraudulent
# transactions". A refused update is a silently dropped learning signal, and it lands on exactly
# the quantity Q3 measures -- an arm losing a third of its updates looks like an arm that cannot
# learn. ACE never hit this because its prompt sees only the complaint, so leaving it unredacted
# would also make the arms incomparable on a dimension unrelated to their recipes.
_SECRET_ARGS = ("access_token", "password", "passwords", "token", "api_key", "secret",
                "authorization", "auth_token", "refresh_token", "credential", "credentials")
# APIs that exist only to authenticate. The preference rules never constrain them (no rule in the
# registry claims a field on login/token endpoints), so dropping the call entirely loses nothing
# a curator could learn from.
_AUTH_APIS = ("login", "logout", "signup", "show_account_passwords", "show_profile")


def _safe_args(arguments: dict) -> dict:
    """Replace credential VALUES with a placeholder, keeping the argument names visible.

    Names are kept because 'the agent passed an access_token' is legitimate trajectory structure;
    it is the 200-character bearer token that reads as exfiltration.
    """
    out = {}
    for k, v in (arguments or {}).items():
        if any(s in str(k).lower() for s in _SECRET_ARGS):
            out[k] = "<redacted>"
        else:
            out[k] = v
    return out


def _actions(ep: EpisodeRecord, limit: int = 40) -> str:
    """The trajectory summary every recipe here reflects over, in one shared format.

    Auth calls are dropped and credential values redacted -- see _SECRET_ARGS. What survives is
    the app.api plus the arguments a preference can actually constrain (message bodies, payment
    notes, privacy flags), which is all any of these recipes needs to induce a preference from.
    """
    lines = []
    for c in ep.api_calls:
        if c.api in _AUTH_APIS:
            continue
        lines.append(f"- {c.app}.{c.api} {_safe_args(c.arguments)}"[:200])
        if len(lines) >= limit:
            break
    return "\n".join(lines) or "(no API calls)"


REFLEXION_PROMPT = """You attempted a task for a user and the user was not satisfied.

What you did:
{actions}

The user's reaction:
{feedback}

Write a short self-reflection that would help you do better next time. State the lesson as
advice to yourself, in one or two sentences. Output ONLY the reflection.
"""


class ReflexionUpdater(BaseUpdater):
    """Reflexion (Shinn et al., arXiv:2303.11366, NeurIPS 2023): verbal self-reflection buffer.

    Stores self-generated natural-language critiques in an episodic buffer and injects the whole
    buffer into the next attempt. Two properties distinguish it from every other arm here:
    the update is **pure append** -- a past reflection is never edited or removed -- and the
    capacity is a **hard sliding window** of the most recent W, older entries simply falling out
    (the paper uses 1-3; ALFWorld and HotpotQA use 3, programming uses 1). No retrieval: the
    windowed buffer is injected in full. Trigger is per-trial on failure, which the complaint
    already is.

    This is the informative arm for a plateau claim because its ceiling is PREDICTABLE rather
    than emergent: a FIFO window of W lines cannot hold more than W preferences at once, so once
    the number of learnable preferences exceeds W the descent must stall for a reason that is
    structural and has nothing to do with what the feedback channel can express. p13 has 3
    carriable preferences against the paper's W=3, so the window is exactly at its limit.

    DEVIATION: the paper's ALFWorld variant classifies failure with a heuristic or an LLM judge;
    here `feedback.accepted` is the failure signal, which is the same bit obtained more cheaply
    and without a second model.
    """

    def __init__(self, llm: BaseLLM, window: int = 3):
        self.llm = llm
        self.window = window
        self.reflections: list[str] = []
        self.updates = 0
        self.dropped = 0          # how many fell out of the window: the arm's own capacity audit
        self.history: list[dict] = []

    def observe(self, ep: EpisodeRecord, feedback: Feedback):
        if feedback.accepted or not feedback.text:
            return
        self.updates += 1
        text = self.llm.generate(
            system="You are a reflective agent improving through self-critique.",
            messages=[{"role": "user", "content": REFLEXION_PROMPT.format(
                actions=_actions(ep), feedback=feedback.text)}],
            max_tokens=256, temperature=0.7,
        ).strip()
        if not text:
            return
        # Append-only, then evict the oldest. Never edit: that is the definitional property of
        # this recipe and the reason it can forget a preference it already learned.
        self.reflections.append(text)
        if len(self.reflections) > self.window:
            self.dropped += len(self.reflections) - self.window
            self.reflections = self.reflections[-self.window:]
        self.history.append({"episode": self.updates, "added": text[:200],
                             "buffer": len(self.reflections), "dropped_total": self.dropped})

    def render_memory(self) -> str:
        if not self.reflections:
            return ""
        return "\n".join(f"- {r}" for r in self.reflections)

    def state(self) -> dict:
        text = self.render_memory()
        return {"reflections": len(self.reflections), "memory_chars": len(text),
                "memory_lines": len(text.splitlines()) if self.reflections else 0,
                "updates": self.updates, "dropped_by_window": self.dropped,
                "window": self.window}

    def save(self, path: str) -> None:
        Path(path).write_text(json.dumps(
            {"reflections": self.reflections, "state": self.state(),
             "history": self.history}, ensure_ascii=False, indent=1))


CHEATSHEET_PROMPT = """You maintain a "cheatsheet" of what you have learned about this user.

Current cheatsheet:
{memory}

What you just did:
{actions}

The user's reaction:
{feedback}

Produce the NEXT version of the cheatsheet. Judge each existing entry for whether it is useful
and generally applicable: refine it, or drop it if it is not earning its place. Add what this
episode taught you. Consolidate overlapping entries so the sheet stays compact. Preserve entries
that are still correct -- do not drop a lesson merely because this episode was about something
else. Output ONLY the new cheatsheet.
"""


class DynamicCheatsheetUpdater(BaseUpdater):
    """Dynamic Cheatsheet, curator variant (Suzgun et al., arXiv:2504.07952, EACL 2025).

    M_{i+1} = Cur(M_i, x_i, y_i): a curator LLM regenerates the whole sheet each round, judging
    each entry's usefulness and generalisability, refining or removing it, and consolidating for
    compactness. Capacity is **soft** -- the paper imposes no entry or token cap, only a
    prompt-level instruction to stay concise, and reports drift from 494 to 1831 tokens on AIME.
    Self-assessment only: no ground-truth labels and no human feedback, which is why it survives
    this experiment's supervision constraints where GEPA and ReasoningBank do not.

    It is the intermediate point between ACE and the naive rewrite: like the naive arm it
    regenerates everything, but unlike it the prompt explicitly asks for preservation and
    consolidation. The paper reports its own failure mode -- the curator abbreviates instead of
    restating, so a stored heuristic degrades over rounds -- which is a *graded* version of the
    catastrophic forgetting the naive arm shows. That makes it a genuinely different cell rather
    than a duplicate of either neighbour.

    DEVIATION: this is DC-Cu (cumulative curation), not DC-RS. The retrieval-synthesis variant
    embeds and retrieves top-k=3 before answering, which would add a retrieval mechanism no other
    arm here has and confound the comparison.
    """

    def __init__(self, llm: BaseLLM, max_memory_chars: int = 4000):
        self.llm = llm
        self.memory = ""
        self.max_memory_chars = max_memory_chars
        self.updates = 0
        self.history: list[dict] = []

    def observe(self, ep: EpisodeRecord, feedback: Feedback):
        if feedback.accepted or not feedback.text:
            return
        self.updates += 1
        before = self.memory
        self.memory = self.llm.generate(
            system="You are a curator maintaining a compact, reusable cheatsheet.",
            messages=[{"role": "user", "content": CHEATSHEET_PROMPT.format(
                memory=self.memory or "(empty)", actions=_actions(ep),
                feedback=feedback.text)}],
            max_tokens=1024, temperature=0.7,
        ).strip()[: self.max_memory_chars]
        # Length per round is the arm's own measurement: the paper's documented failure is the
        # sheet shrinking as the curator abbreviates, so growth has to be on record rather than
        # inferred from the violation curve.
        self.history.append({"episode": self.updates, "chars_before": len(before),
                             "chars_after": len(self.memory),
                             "lines_after": len([x for x in self.memory.splitlines() if x.strip()])})

    def render_memory(self) -> str:
        return self.memory

    def state(self) -> dict:
        lines = [x for x in self.memory.splitlines() if x.strip()]
        return {"memory_chars": len(self.memory), "memory_lines": len(lines),
                "updates": self.updates,
                "truncated": len(self.memory) >= self.max_memory_chars}

    def save(self, path: str) -> None:
        Path(path).write_text(json.dumps(
            {"memory": self.memory, "state": self.state(), "history": self.history},
            ensure_ascii=False, indent=1))


AMEM_NOTE_PROMPT = """Write a memory note about what this interaction taught you about the user.

What you did:
{actions}

The user's reaction:
{feedback}

Output ONLY valid JSON:
{{"content": "the lesson, one sentence",
  "keywords": ["3-5", "short", "keywords"],
  "context": "one sentence on when this applies"}}
"""

AMEM_EVOLVE_PROMPT = """A new memory note was just created:
  {new}

It resembles this existing note:
  {old}

Should the existing note's framing be updated in light of the new one? Its CONTENT must not
change -- only how it is contextualised. Output ONLY valid JSON:
{{"update": true or false, "context": "revised context sentence", "keywords": ["..."]}}
"""


@dataclass
class MemNote:
    """One A-MEM note. `content` is immutable after creation by construction of the recipe."""
    id: str
    content: str
    keywords: list[str] = field(default_factory=list)
    context: str = ""
    links: list[str] = field(default_factory=list)


class AMemUpdater(BaseUpdater):
    """A-MEM (Xu et al., arXiv:2502.12110, NeurIPS 2025): agentic memory with note evolution.

    A note is (content, keywords, context, embedding, links). On each new note, cosine similarity
    picks the top-k nearest existing notes, links them, and then -- the distinctive step -- an LLM
    may rewrite each NEIGHBOUR's context and keywords in light of the newcomer, while that
    neighbour's `content` stays fixed. Capacity is **unbounded**: the paper explicitly contrasts
    itself with Ebbinghaus-decay approaches and has no eviction or deletion at all.

    So it fills the cell nothing else here occupies -- unbounded growth with delta-granular edits.
    That combination is what makes it diagnostic: if the plateau were caused by ACE's fixed
    capacity, an arm that never evicts anything should escape it. If this arm plateaus too, the
    capacity explanation is dead and the feedback channel is the remaining candidate.

    DEVIATIONS, both forced and both recorded because they change what the arm measures:
    (1) The paper's prose describes strengthen/update-neighbour while its JSON example also lists
        "merge" and "prune", which are never specified. Only neighbour-update is implemented, and
        no note is ever deleted -- the reading consistent with the paper's own no-eviction claim.
    (2) Retrieval is flat top-k over `all-minilm-l6-v2` in the paper. Here the whole note set is
        injected (the run reaches ~7 notes, far below any retrieval budget) and similarity uses
        the same char-trigram pseudo-embedding as ACE's dedup, so the two arms share one
        similarity function and neither gains from a better encoder the other lacks.
    """

    def __init__(self, llm: BaseLLM, top_k: int = 3, link_threshold: float = 0.55):
        self.llm = llm
        self.top_k = top_k
        self.link_threshold = link_threshold
        self.notes: list[MemNote] = []
        self.embeddings: dict[str, list[float]] = {}
        self.updates = 0
        self.parse_failures = 0
        self.evolved = 0          # neighbour context rewrites actually applied
        self.history: list[dict] = []

    def _write_note(self, ep: EpisodeRecord, feedback: Feedback) -> MemNote | None:
        raw = self.llm.generate(
            system="You are an agentic memory system.",
            messages=[{"role": "user", "content": AMEM_NOTE_PROMPT.format(
                actions=_actions(ep), feedback=feedback.text)}],
            max_tokens=384, temperature=0.7,
        ).strip()
        parsed = _extract_json_array(raw)
        obj = parsed[0] if isinstance(parsed, list) and parsed else None
        if not isinstance(obj, dict):
            return None
        content = str(obj.get("content", "") or "").strip()
        if not content:
            return None
        kw = [str(k).strip() for k in (obj.get("keywords") or []) if str(k).strip()]
        return MemNote(id=hashlib.md5(content.encode()).hexdigest()[:6], content=content,
                       keywords=kw[:5], context=str(obj.get("context", "") or "").strip())

    def observe(self, ep: EpisodeRecord, feedback: Feedback):
        if feedback.accepted or not feedback.text:
            return
        self.updates += 1
        note = self._write_note(ep, feedback)
        if note is None:
            self.parse_failures += 1
            self.history.append({"episode": self.updates, "parse_failed": True})
            return
        if any(n.id == note.id for n in self.notes):
            self.history.append({"episode": self.updates, "duplicate_id": note.id})
            return

        # Embed over content + keywords + context, as the paper does, so a note's framing
        # participates in its own similarity rather than only its content.
        emb = embed_text(" ".join([note.content, " ".join(note.keywords), note.context]))
        scored = sorted(((cosine(emb, self.embeddings[n.id]), n) for n in self.notes
                         if n.id in self.embeddings), key=lambda t: -t[0])
        neighbours = [(s, n) for s, n in scored[: self.top_k] if s >= self.link_threshold]

        evolved_here = []
        for _score, nb in neighbours:
            note.links.append(nb.id)
            nb.links.append(note.id)
            raw = self.llm.generate(
                system="You are an agentic memory system deciding whether to recontextualise.",
                messages=[{"role": "user", "content": AMEM_EVOLVE_PROMPT.format(
                    new=note.content, old=f"{nb.content} (context: {nb.context})")}],
                max_tokens=256, temperature=0.7,
            ).strip()
            parsed = _extract_json_array(raw)
            obj = parsed[0] if isinstance(parsed, list) and parsed else None
            if not isinstance(obj, dict) or not obj.get("update"):
                continue
            ctx = str(obj.get("context", "") or "").strip()
            if ctx:
                nb.context = ctx
            kw = [str(k).strip() for k in (obj.get("keywords") or []) if str(k).strip()]
            if kw:
                nb.keywords = kw[:5]
            # The neighbour's framing changed, so its embedding is stale unless recomputed --
            # the paper does not say, and leaving it stale would make later similarity scores
            # refer to text no longer stored.
            self.embeddings[nb.id] = embed_text(
                " ".join([nb.content, " ".join(nb.keywords), nb.context]))
            self.evolved += 1
            evolved_here.append(nb.id)

        self.notes.append(note)
        self.embeddings[note.id] = emb
        self.history.append({"episode": self.updates, "added": note.id,
                             "content": note.content[:160],
                             "linked": [n.id for _s, n in neighbours],
                             "evolved": evolved_here, "notes_after": len(self.notes)})

    def render_memory(self) -> str:
        if not self.notes:
            return ""
        # context is appended when present: it is where this recipe's evolution step lands, so
        # omitting it would inject a memory in which neighbour-update has no observable effect.
        out = []
        for n in self.notes:
            line = f"- [{n.id}] {n.content}"
            if n.context:
                line += f" ({n.context})"
            out.append(line)
        return "\n".join(out)

    def state(self) -> dict:
        text = self.render_memory()
        return {"notes": len(self.notes), "memory_chars": len(text),
                "memory_lines": len(text.splitlines()) if self.notes else 0,
                "updates": self.updates, "evolved": self.evolved,
                "parse_failures": self.parse_failures,
                "links": sum(len(n.links) for n in self.notes) // 2}

    def save(self, path: str) -> None:
        Path(path).write_text(json.dumps(
            {"notes": [asdict(n) for n in self.notes], "state": self.state(),
             "history": self.history}, ensure_ascii=False, indent=1))


TEPA_WRITE_PROMPT = """Record what this interaction taught you about the user, as keyed precedents.

What you did:
{actions}

The user's reaction:
{feedback}

The KEY names which attribute of which artefact the preference constrains. Choose the key from
this fixed list, using the closest match:
{keys}

Use `other.<slug>` only if no key above could apply. Two precedents about the same attribute must
get the SAME key even if worded differently, because a newer precedent REPLACES the active one
under its key.

The reaction may contain SEVERAL complaints. Emit one precedent per distinct complaint.

Output ONLY a valid JSON array, no prose before or after:
[{{"key": "artefact.attribute", "content": "the preference as one sentence"}}]
"""

# Fixed key vocabulary. Replacement is a within-key operation, so it only works when the same
# preference lands on the same key every time -- and left to invent slugs freely, the writer spread
# ONE preference across four keys on seed 0 (payment_note.category_tag, .prefix, .format, and a
# stray .suffix), so nothing ever replaced anything. The list is derived from the artefact/aspect
# vocabulary the environment already uses (feedback._ASPECT), not from the persona under test, so
# it names the space of things a preference COULD constrain without revealing which ones are
# scored. `other.<slug>` keeps the writer able to express something outside the list.
TEPA_KEYS = (
    "sms.opening", "sms.ending", "sms.length", "sms.punctuation", "sms.names",
    "payment_note.content", "payment_note.prefix", "payment_note.ending",
    "payment_note.case", "payment_note.length",
    "payment.visibility", "payment.card",
    "playlist.visibility", "music.save_method",
    "file.name", "file.content_opening",
    "email.opening", "email.ending", "email.subject",
    "note.title", "note.tags", "task.due_date", "task.title",
)

@dataclass
class Precedent:
    """One keyed precedent. `active` is the validity state TEPA makes explicit."""
    id: str
    key: str
    content: str
    active: bool = True
    added_at: int = 0
    revoked_at: int | None = None
    revoked_by: str = ""


class TepaUpdater(BaseUpdater):
    """TEPA (arXiv:2608.07429v2, Aug 2026): revocable evidence-memory with keyed precedents.

    Observations are stored as precedents carrying a KEY (which attribute they constrain) plus an
    explicit validity state. The newest precedent under a key **revokes** whatever was active under
    it -- the paper's "current-key replacement", which it names as the decisive operation and
    confirms by matching a last-write-wins cache on SH-6k. A revoked precedent leaves the injected
    memory but stays in the store for audit, and is re-promoted if identical evidence reappears.

    This is a third capacity regime, outside the fixed-vs-unbounded axis the other arms span:
    storage is unbounded, while the ACTIVE set is bounded by the number of distinct keys. None of
    the other arms can retract a past assertion -- ACE can overwrite a bullet, Reflexion can only
    let one fall out of its window, and a rewrite arm can only fail to restate it. Retraction with
    an audit trail is a distinct operation.

    Why it is the most informative addition here: the paper's own headline experiment is a
    preference-update stream, and it reports append-only and last-write-wins both falling BELOW
    the no-memory baseline under reversal (0.210 and 0.210 against 0.309) while TEPA holds 0.950.
    That is a direct, falsifiable prediction about the regime this experiment measures.

    DEVIATIONS:
    (1) The paper's retrieval draws from the active set; here the whole active set is injected,
        because it stays small (bounded by the number of distinct keys) and every other arm also
        injects its whole memory. Adding retrieval to one arm only would confound the comparison.
    (2) Keys are chosen from a FIXED vocabulary (TEPA_KEYS) rather than invented per episode. The
        paper assumes keys arrive with the observation schema, which a free-text preference stream
        does not provide, and an unconstrained writer spread one preference over four keys on the
        first run -- which disabled replacement entirely and silently turned this into the
        append-only baseline the paper reports failing. `key_snaps` in state() records how often
        the vocabulary had to correct the writer.
    """

    def __init__(self, llm: BaseLLM):
        self.llm = llm
        self.precedents: list[Precedent] = []
        self.updates = 0
        self.parse_failures = 0
        self.revocations = 0
        self.repromotions = 0
        # How often the writer's key had to be snapped onto TEPA_KEYS. Recorded because a high
        # count means the fixed vocabulary is doing the work rather than the writer, which changes
        # how much of this arm's behaviour is attributable to the recipe.
        self.key_snaps = 0
        self.history: list[dict] = []
        self._last_raw = ""      # last writer reply, kept only for the parse-failure audit

    def _write(self, ep: EpisodeRecord, feedback: Feedback) -> list[tuple[str, str]]:
        """-> [(key, content)] for every precedent this episode's complaints support.

        Returns a LIST, not one pair. The instance tier emits one line per violated rule and a
        single episode here routinely carries five, so taking only the first would silently
        discard four fifths of the evidence and make the arm look like it learns nothing. That is
        exactly how the first smoke run failed.

        max_tokens is generous for the same reason: five precedents run to ~500 characters, and a
        truncated array is unparseable, which the audit trail would report as a parse failure
        indistinguishable from a model that refused to answer.
        """
        raw = self.llm.generate(
            system="You maintain a revocable evidence memory. Output only JSON.",
            messages=[{"role": "user", "content": TEPA_WRITE_PROMPT.format(
                actions=_actions(ep), feedback=feedback.text,
                keys="\n".join(f"  {k}" for k in TEPA_KEYS))}],
            max_tokens=1024, temperature=0.7,
        ).strip()
        self._last_raw = raw
        parsed = _extract_json_array(raw)
        if not isinstance(parsed, list):
            return []
        out = []
        for obj in parsed:
            if not isinstance(obj, dict):
                continue
            key = str(obj.get("key", "") or "").strip().lower()
            content = str(obj.get("content", "") or "").strip()
            if not (key and content):
                continue
            # Snap a near-miss onto the vocabulary so replacement still fires. Left unsnapped, a
            # writer that answers 'payment_note.category_tag' creates a key nothing will ever
            # replace, which is how the first run degenerated into append-only.
            if key not in TEPA_KEYS and not key.startswith("other."):
                snapped = next((k for k in TEPA_KEYS if k.split(".")[0] == key.split(".")[0]
                                and k.split(".")[1] in key), "")
                key = snapped or key
                if snapped:
                    self.key_snaps += 1
            out.append((key, content))
        return out

    def observe(self, ep: EpisodeRecord, feedback: Feedback):
        if feedback.accepted or not feedback.text:
            return
        self.updates += 1
        written = self._write(ep, feedback)
        if not written:
            # Record the raw reply: a plateau caused by the writer answering conversationally is a
            # bug in this file, and without the text the audit cannot tell that from a model that
            # had nothing to say. The first smoke run needed exactly this to be diagnosed.
            self.parse_failures += 1
            self.history.append({"episode": self.updates, "parse_failed": True,
                                 "raw": self._last_raw[:200]})
            return
        # One complaint line per violated rule, so one episode yields several precedents. Each is
        # applied independently: a later one may revoke an earlier one under the same key, which is
        # the mechanism doing its job rather than a conflict to suppress.
        for key, content in written:
            self._apply_one(key, content)

    def _revoke_key(self, key: str, except_id: str, by: str) -> list[str]:
        """Deactivate every active precedent under `key` except `except_id`. -> revoked ids.

        The single place the one-active-per-key invariant is enforced, so a new write and a
        re-promotion cannot diverge on it.
        """
        revoked = []
        for p in self.precedents:
            if p.active and p.key == key and p.id != except_id:
                p.active = False
                p.revoked_at = self.updates
                p.revoked_by = by
                self.revocations += 1
                revoked.append(p.id)
        return revoked

    def _apply_one(self, key: str, content: str) -> None:
        new_id = hashlib.md5(f"{key}|{content}".encode()).hexdigest()[:6]

        # Re-promotion: identical evidence reappearing means the revocation was premature, and the
        # paper keeps revoked history precisely so this is possible. It is still a current-key
        # write, so it must revoke whatever else is active under the key -- otherwise the key ends
        # up with two active precedents and the "one active per key" invariant this class documents
        # silently breaks. Seed 1 ended with active=6 over keys=5 for exactly this reason: episode
        # 17 wrote a new precedent under payment_note.ending, then episode 18 re-promoted an older
        # one under the same key and both stayed active.
        for p in self.precedents:
            if p.id == new_id and not p.active:
                p.active, p.revoked_at, p.revoked_by = True, None, ""
                self.repromotions += 1
                superseded = self._revoke_key(key, except_id=new_id, by=new_id)
                self.history.append({"episode": self.updates, "repromoted": p.id, "key": key,
                                     "revoked": superseded,
                                     "active_after": sum(1 for q in self.precedents if q.active)})
                return
        if any(p.id == new_id and p.active for p in self.precedents):
            self.history.append({"episode": self.updates, "duplicate": new_id, "key": key})
            return

        # CURRENT-KEY REPLACEMENT: the newest precedent under a key supersedes whatever was active
        # under it, so the active set holds at most one precedent per key. The paper names this
        # ("current-key replacement is the decisive operation") and its own SH-6k result is that
        # TEPA matches a last-write-wins cache, which is only possible under replacement.
        #
        # An earlier version gated this on an LLM contradiction check, and that was wrong twice
        # over. It fired 0 times in 18 updates on seed 0 because the writer restates a preference
        # rather than negating it ("must include a category tag" then "must include a category tag
        # (e.g. 'Groceries')"), which is not a contradiction; the active set grew to 23 lines over
        # 9 keys and the arm silently degenerated into append-only -- the very baseline the paper
        # reports falling below no-memory. It also contradicted this class's own docstring, which
        # claims the active set is bounded by key uniqueness.
        revoked_here = self._revoke_key(key, except_id=new_id, by=new_id)

        self.precedents.append(Precedent(id=new_id, key=key, content=content,
                                         added_at=self.updates))
        self.history.append({"episode": self.updates, "added": new_id, "key": key,
                             "content": content[:160], "revoked": revoked_here,
                             "active_after": sum(1 for p in self.precedents if p.active),
                             "stored_after": len(self.precedents)})

    def render_memory(self) -> str:
        """Active precedents only. Revoked ones stay in the store and out of the prompt -- that
        separation IS the mechanism, so rendering them would silently turn this into append-only.
        """
        active = [p for p in self.precedents if p.active]
        if not active:
            return ""
        return "\n".join(f"- [{p.key}] {p.content}" for p in active)

    def state(self) -> dict:
        text = self.render_memory()
        active = [p for p in self.precedents if p.active]
        return {"active": len(active), "stored": len(self.precedents),
                "keys": len({p.key for p in active}),
                "memory_chars": len(text),
                "memory_lines": len(text.splitlines()) if active else 0,
                "updates": self.updates, "revocations": self.revocations,
                "repromotions": self.repromotions, "key_snaps": self.key_snaps,
                "parse_failures": self.parse_failures}

    def save(self, path: str) -> None:
        Path(path).write_text(json.dumps(
            {"precedents": [asdict(p) for p in self.precedents], "state": self.state(),
             "history": self.history}, ensure_ascii=False, indent=1))


# TRACE compiles each correction into a CHECK, not a sentence. The check has to be executable
# against an episode, so the learner emits a constrained predicate rather than free code: a
# generated `eval` would be arbitrary code execution driven by model output, and a generated
# regex would let one malformed rule reject every episode forever. The four predicates cover the
# forms the complaints in this environment actually take (prefix, suffix, contains, equals) over
# the argument of one API.
TRACE_COMPILE_PROMPT = """The user corrected you. Turn each correction into an executable check.

What you did:
{actions}

The user's correction:
{feedback}

A check is a JSON object with these fields:
  "api"       which call it applies to, e.g. "phone.send_text_message", "venmo.create_transaction"
  "argument"  which argument of that call to inspect. Use EXACTLY one of the names that appears in
              the trajectory above for that call -- a name that does not appear is never checked.
  "predicate" one of: "startswith", "endswith", "contains", "equals",
              "not_startswith", "not_endswith", "not_contains"
  "value"     the literal string the argument must (or must not) start with / end with / contain
  "rule"      the correction restated as one imperative sentence, for the user-facing memory

The "value" must be a GENERAL requirement, never a quotation from this one episode's text. Compile
"sign texts with my first name" as endswith the name, NOT as a check on the whole message you
happened to send. A check whose value is copied out of this episode's content can never pass on a
different task, so it would block every future episode.

Use a "not_" predicate for a correction that forbids something. Do not express a prohibition as a
positive check on the forbidden text.

The correction may contain SEVERAL complaints. Emit one check per complaint that names something
concrete enough to test mechanically; for a complaint that does not, emit {{"skip": true}}.

Output ONLY a valid JSON array, no prose before or after.
"""

_TRACE_PREDICATES = {
    "startswith": lambda got, want: got.lower().startswith(want.lower()),
    "endswith": lambda got, want: got.lower().rstrip(".!?").endswith(want.lower()),
    "contains": lambda got, want: want.lower() in got.lower(),
    "equals": lambda got, want: got.strip().lower() == want.strip().lower(),
    # Negations exist because without them the writer inverts a prohibition into a positive check.
    # Seed 0 compiled "Do not start SMS messages with 'lie, The Godfather: Part II, The Pianist'"
    # as startswith that string, i.e. a rule demanding the very text it was told to avoid, which
    # then failed 25/25 times and could never pass.
    "not_startswith": lambda got, want: not got.lower().startswith(want.lower()),
    "not_endswith": lambda got, want: not got.lower().rstrip(".!?").endswith(want.lower()),
    "not_contains": lambda got, want: want.lower() not in got.lower(),
}

# A compiled value longer than this is almost certainly quoted episode content rather than a
# general requirement. Seed 0 produced startswith 'ndations: Django Unchained, Pulp Fiction' --
# a fragment of one task's own message body, mid-word at both ends, which no other episode can
# satisfy. Such a rule is a permanent gate that blocks every future episode, so it is rejected at
# compile time and counted rather than silently admitted.
_TRACE_MAX_VALUE_LEN = 24


@dataclass
class TraceRule:
    """One compiled check plus its natural-language restatement."""
    id: str
    api: str
    argument: str
    predicate: str
    value: str
    rule: str
    fired: int = 0        # how many times it was evaluated
    failed: int = 0       # how many times it rejected


class TraceUpdater(BaseUpdater):
    """TRACE (arXiv:2606.13174, Jun 2026): mine corrections, compile them into runtime checks.

    Mines the user's own corrections, rewrites each as an atomic rule, and compiles it into a
    runtime check that must pass before the agent may declare the task complete. Its input signal
    is literally free-form user corrections, which is why it needs no adaptation to this
    experiment's supervision; the paper's baseline comparison is Mem0, which still leaves 57.5%
    of applicable preference checks violated.

    What it adds beyond the capacity x granularity 2x2: **self-verification at action time**. Every
    other arm here only changes what the agent is TOLD; this one changes what the agent is
    ALLOWED to finish. `render_memory` states the rules and, separately, `check(ep)` evaluates the
    compiled predicates so the harness can enforce them.

    A known interaction worth stating rather than discovering later: a compiled gate on a
    mechanically unsatisfiable preference blocks forever instead of degrading. That case exists in
    this codebase (`sms_terse` on S1, whose eval tasks dictate a body already over the word cap),
    so `check` reports which rules failed and lets the caller decide, rather than looping.

    DEVIATIONS:
    (1) The compiled check is a constrained predicate over one API argument, not generated code.
        Executing model-authored code is not acceptable here, and an unconstrained regex would let
        one malformed rule reject every subsequent episode.
    (2) The paper targets a coding-agent runtime whose checks can call the interpreter. Here the
        check runs against the episode's recorded API calls, which is the only observable this
        harness exposes.
    """

    def __init__(self, llm: BaseLLM, max_rules: int = 12):
        self.llm = llm
        self.rules: list[TraceRule] = []
        self.max_rules = max_rules
        self.updates = 0
        self.parse_failures = 0
        self.skipped = 0          # corrections the learner judged untestable
        # Checks rejected for quoting episode content. Recorded because each one would have been a
        # permanent gate, and the count says how often the compiler reaches for episode text.
        self.overlong = 0
        # Checks skipped because the call did not carry the named argument. A high count means the
        # compiler is guessing argument names, which is a harness-addressing problem rather than
        # anything about the preference.
        self.inapplicable = 0
        self.history: list[dict] = []

    def observe(self, ep: EpisodeRecord, feedback: Feedback):
        if feedback.accepted or not feedback.text:
            return
        self.updates += 1
        raw = self.llm.generate(
            system="You compile user corrections into executable checks. Output only JSON.",
            messages=[{"role": "user", "content": TRACE_COMPILE_PROMPT.format(
                actions=_actions(ep), feedback=feedback.text)}],
            max_tokens=1024, temperature=0.7,
        ).strip()
        parsed = _extract_json_array(raw)
        if not isinstance(parsed, list) or not parsed:
            self.parse_failures += 1
            self.history.append({"episode": self.updates, "parse_failed": True,
                                 "raw": raw[:200]})
            return
        # One check per complaint line, for the same reason TEPA writes several precedents: the
        # instance tier reports every violated rule, so compiling only the first would discard
        # most of the corrections the paper's whole method is built on mining.
        for obj in parsed:
            if isinstance(obj, dict):
                self._compile_one(obj, raw)

    def _compile_one(self, obj: dict, raw: str) -> None:
        if obj.get("skip"):
            self.skipped += 1
            self.history.append({"episode": self.updates, "skipped": True})
            return
        pred = str(obj.get("predicate", "") or "").strip().lower()
        api = str(obj.get("api", "") or "").strip()
        argument = str(obj.get("argument", "") or "").strip()
        value = str(obj.get("value", "") or "")
        rule = str(obj.get("rule", "") or "").strip()
        if pred not in _TRACE_PREDICATES or not api or not argument or not rule:
            self.parse_failures += 1
            self.history.append({"episode": self.updates, "parse_failed": True,
                                 "reason": "unusable check", "raw": raw[:200]})
            return
        # Reject episode-quoted values: a check whose literal is a fragment of this task's own text
        # can never be satisfied by a different task, so admitting it installs a permanent gate.
        # `equals` is exempt because an exact-match requirement legitimately carries a full value.
        if pred != "equals" and len(value) > _TRACE_MAX_VALUE_LEN:
            self.overlong += 1
            self.history.append({"episode": self.updates, "rejected": "value too long",
                                 "predicate": pred, "value": value[:60]})
            return
        rid = hashlib.md5(f"{api}|{argument}|{pred}|{value}".encode()).hexdigest()[:6]
        if any(r.id == rid for r in self.rules):
            self.history.append({"episode": self.updates, "duplicate": rid})
            return
        self.rules.append(TraceRule(id=rid, api=api, argument=argument, predicate=pred,
                                    value=value, rule=rule))
        dropped = 0
        if len(self.rules) > self.max_rules:
            dropped = len(self.rules) - self.max_rules
            self.rules = self.rules[-self.max_rules:]
        self.history.append({"episode": self.updates, "compiled": rid, "api": api,
                             "argument": argument, "predicate": pred, "value": value[:60],
                             "rule": rule[:160], "dropped": dropped,
                             "rules_after": len(self.rules)})

    def check(self, ep: EpisodeRecord) -> list[str]:
        """Evaluate the compiled checks against an episode -> messages for the rules that failed.

        Only calls matching a rule's api are inspected, and a rule whose api never appears is
        neither passed nor failed -- it is inapplicable, the same convention the persona checkers
        use, so an episode is never rejected for not having done something the rule is silent on.

        A rule naming an argument the call does not carry is likewise inapplicable rather than
        failed. Without this, a compiler that guesses the argument name wrong produces a check that
        fails on EVERY episode for a reason unrelated to the preference: seed 0 compiled
        `is_private equals true` against venmo's actual `private` argument and it failed 65 of 68
        times on the missing key alone, which reads as the arm regressing when it is the harness
        mis-addressing the field.
        """
        failures = []
        for r in self.rules:
            app, _, api = r.api.partition(".")
            calls = [c for c in ep.api_calls
                     if c.app == app and (not api or c.api == api)]
            if not calls:
                continue
            present = [c for c in calls if r.argument in (c.arguments or {})]
            if not present:
                self.inapplicable += 1
                continue
            r.fired += 1
            fn = _TRACE_PREDICATES[r.predicate]
            for c in present:
                got = str(c.arg(r.argument, default=""))
                if not fn(got, r.value):
                    r.failed += 1
                    failures.append(f"{r.rule} (check: {r.argument} must "
                                    f"{r.predicate} {r.value!r}, got {got[:60]!r})")
                    break
        return failures

    def render_memory(self) -> str:
        if not self.rules:
            return ""
        return "\n".join(f"- {r.rule}" for r in self.rules)

    def state(self) -> dict:
        text = self.render_memory()
        return {"rules": len(self.rules), "memory_chars": len(text),
                "memory_lines": len(text.splitlines()) if self.rules else 0,
                "updates": self.updates, "skipped": self.skipped,
                "overlong_rejected": self.overlong,
                "arg_inapplicable": self.inapplicable,
                "parse_failures": self.parse_failures,
                "checks_fired": sum(r.fired for r in self.rules),
                "checks_failed": sum(r.failed for r in self.rules)}

    def save(self, path: str) -> None:
        Path(path).write_text(json.dumps(
            {"rules": [asdict(r) for r in self.rules], "state": self.state(),
             "history": self.history}, ensure_ascii=False, indent=1))
