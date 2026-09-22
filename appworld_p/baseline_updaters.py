"""Q3 adaptations of Reflexion, TEPA, and TRACE.

Reflexion: arXiv:2303.11366; TEPA: arXiv:2608.07429; TRACE: arXiv:2606.13174.
Trajectory inputs omit authentication calls and redact credential values.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from .ace_updater import _extract_json_array
from .episode import EpisodeRecord
from .feedback import Feedback
from .llm import BaseLLM
from .updaters import BaseUpdater


_SECRET_ARGS = ("access_token", "password", "passwords", "token", "api_key", "secret",
                "authorization", "auth_token", "refresh_token", "credential", "credentials")


_AUTH_APIS = ("login", "logout", "signup", "show_account_passwords", "show_profile")


def _safe_args(arguments: dict) -> dict:
    """Replace credential values with a placeholder, preserving argument names."""
    out = {}
    for k, v in (arguments or {}).items():
        if any(s in str(k).lower() for s in _SECRET_ARGS):
            out[k] = "<redacted>"
        else:
            out[k] = v
    return out


def _actions(ep: EpisodeRecord, limit: int = 40) -> str:
    """Summarize API calls after removing authentication calls and redacting credentials."""
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
    """Reflexion adaptation (arXiv:2303.11366).
    Append one reflection per rejected episode and retain the latest window entries.
    The default window is three; each entry may describe multiple preferences."""

    def __init__(self, llm: BaseLLM, window: int = 3):
        self.llm = llm
        self.window = window
        self.reflections: list[str] = []
        self.updates = 0
        self.dropped = 0
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
    """TEPA adaptation (arXiv:2608.07429).
    Maintain one active precedent per key and archive revoked entries.
    Inject the entire active set rather than retrieving a subset.
    The prompt specifies TEPA_KEYS and permits other.* keys; recognized key variants
    are mapped back to the vocabulary. Parsing and replacement diagnostics are recorded."""

    def __init__(self, llm: BaseLLM):
        self.llm = llm
        self.precedents: list[Precedent] = []
        self.updates = 0
        self.parse_failures = 0
        self.revocations = 0
        self.repromotions = 0

        self.key_snaps = 0
        self.history: list[dict] = []
        self._last_raw = ""

    def _write(self, ep: EpisodeRecord, feedback: Feedback) -> list[tuple[str, str]]:
        """Return keyed precedents for all supported complaints in an episode."""
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
            self.parse_failures += 1
            self.history.append({"episode": self.updates, "parse_failed": True,
                                 "raw": self._last_raw[:200]})
            return

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

        revoked_here = self._revoke_key(key, except_id=new_id, by=new_id)

        self.precedents.append(Precedent(id=new_id, key=key, content=content,
                                         added_at=self.updates))
        self.history.append({"episode": self.updates, "added": new_id, "key": key,
                             "content": content[:160], "revoked": revoked_here,
                             "active_after": sum(1 for p in self.precedents if p.active),
                             "stored_after": len(self.precedents)})

    def render_memory(self) -> str:
        """Render active precedents; keep revoked entries in the archive only."""
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

    "not_startswith": lambda got, want: not got.lower().startswith(want.lower()),
    "not_endswith": lambda got, want: not got.lower().rstrip(".!?").endswith(want.lower()),
    "not_contains": lambda got, want: want.lower() not in got.lower(),
}


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
    fired: int = 0
    failed: int = 0


class TraceUpdater(BaseUpdater):
    """TRACE adaptation (arXiv:2606.13174).
    Compile feedback into constrained predicates over recorded API arguments.
    Supported predicates, value restrictions, and the key schema are defined below.
    The driver retries against these learned checks, not the persona checker.
    A missing API or argument is inapplicable. Capacity defaults to 12 checks."""

    def __init__(self, llm: BaseLLM, max_rules: int = 12):
        self.llm = llm
        self.rules: list[TraceRule] = []
        self.max_rules = max_rules
        self.updates = 0
        self.parse_failures = 0
        self.skipped = 0

        self.overlong = 0

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
        """Evaluate learned predicates; missing APIs or arguments are inapplicable."""
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
