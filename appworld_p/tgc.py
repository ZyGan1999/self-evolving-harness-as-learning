"""Internal task-completion diagnostics.
Run the official evaluator and separately adjudicate preference-owned text decorations.
These diagnostics are not the paper's reported preference-violation metric."""
from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field


RULE_HOST_PARAM = {
    "private_note_format": "description",
    "payment_has_note": "description",
    "txn_running_count": "description",
    "sms_char_checksum": "message",
    "sms_checksum": "message",
    "sms_signoff": "message",
    "email_greeting_signoff": "body",
    "email_subject_short": "subject",
    "checksum_subject": "subject",
    "note_title_dated": "title",
    "note_tags_required": "tags",
}


ADDITIVE_RULES = frozenset({
    "private_note_format", "payment_has_note", "txn_running_count",
    "sms_char_checksum", "sms_checksum", "sms_signoff",
    "email_greeting_signoff", "note_title_dated", "note_tags_required",
})


_ASSERT_RE = re.compile(r"^AssertionError:[ \t]*(.*?)(?=^Original values:|\Z)",
                        re.MULTILINE | re.DOTALL)


_SMS_CHECKSUM_SUFFIX = re.compile(r"\s*#\s*\d+\s*$")
_PRIVATE_NOTE = re.compile(r"^\s*\[[^|\]]*\|[^|\]]*\|(?P<summary>[^\]]*)\]\s*$")


def _strip_sms_checksum(text: str) -> str:
    return _SMS_CHECKSUM_SUFFIX.sub("", text).strip()


def _strip_private_note(text: str) -> str:
    m = _PRIVATE_NOTE.match(text)
    return m.group("summary").strip() if m else text.strip()


_SPEND_TOTAL_SUFFIX = re.compile(r"[\s,;.]*total:\s*\$?[0-9]+(?:\.[0-9]{1,2})?\s*$", re.I)


def _strip_spend_total(text: str) -> str:
    return _SPEND_TOTAL_SUFFIX.sub("", text).strip()


STRIP_DECORATION = {
    "sms_char_checksum": _strip_sms_checksum,
    "sms_checksum": _strip_sms_checksum,
    "private_note_format": _strip_private_note,
    "txn_running_count": _strip_sms_checksum,
    "spend_total_per_recipient": _strip_spend_total,
    "spend_total_all_time": _strip_spend_total,
}


@dataclass
class Verdict:
    """Adjudication of one episode's official evaluation."""
    official: bool
    relaxed: bool
    excused: list[str] = field(default_factory=list)
    unexcused: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def _split_comparison(block: str) -> tuple[object, object] | None:
    """Parse "<left repr> == <right repr>" out of an AssertionError block.

    Returns None rather than guessing when the block does not parse, so an unparsable trace
    stays a failure instead of being silently excused.
    """
    block = block.strip()
    lines = [l for l in (x.strip() for x in block.splitlines()) if l]
    if "==" in lines:
        i = lines.index("==")
        head, tail = "\n".join(lines[:i]), "\n".join(lines[i + 1:])
    elif " == " in block:
        head, _, tail = block.rpartition(" == ")
    else:
        return None
    try:
        return ast.literal_eval(head.strip()), ast.literal_eval(tail.strip())
    except (ValueError, SyntaxError):
        return None


def _values_from_trace(trace: str) -> tuple[object, object] | None:
    """Actual/expected from a failure trace, preferring the normalized comparison line.

    AppWorld prints the normalized comparison first ("AssertionError: 'done. #5' == 'done.'")
    and then the pre-normalization values under "Original values:". The normalized pair is
    what the official assertion actually compared -- ignore_case/strip already applied -- so
    containment on it matches official semantics rather than re-deriving them.
    """
    for m in _ASSERT_RE.finditer(trace):
        parsed = _split_comparison(m.group(1))
        if parsed is not None:
            return parsed
    return None


def _contains(actual: object, expected: object) -> tuple[bool, str]:
    """Is `expected` still present in `actual` as written? -> (ok, reason-if-not).

    Rejects anything that is not a plain non-empty string comparison, which is what keeps
    amount/id/count assertions out: those never satisfy string containment, and treating
    them as excusable would relax assertions no preference ever touches.
    """
    if not isinstance(expected, str) or not expected.strip():
        return False, f"expected is not a non-empty string ({type(expected).__name__})"
    exp = expected.strip()

    actuals = actual if isinstance(actual, (list, tuple)) else [actual]
    if not actuals:
        return False, "no actual value"
    for a in actuals:
        if not isinstance(a, str):
            return False, f"actual is not a string ({type(a).__name__})"
        if exp not in a.strip():
            return False, f"expected text absent from actual: {a.strip()[:60]!r}"
    return True, ""


def host_params(rule_names) -> dict[str, list[str]]:
    """-> {param token: [rules claiming it]}, restricted to additive rules.

    A rule outside ADDITIVE_RULES contributes nothing: no relaxation may excuse a field the
    preference rewrites or shortens, since there the official expected value is legitimately
    absent from a compliant answer.
    """
    out: dict[str, list[str]] = {}
    for r in rule_names:
        if r in ADDITIVE_RULES and r in RULE_HOST_PARAM:
            out.setdefault(RULE_HOST_PARAM[r], []).append(r)
    return out


def _judge(actual: object, expected: object, rules: list[str]) -> tuple[bool, str]:
    """Would the official assertion have passed with this rule's decoration removed?

    Tries each claiming rule's strip function and re-applies exact equality. Falls back to
    containment only when no rule offers a strip function, flagging it in the reason so the
    weaker basis is visible rather than silent.
    """
    if not isinstance(expected, str) or not expected.strip():
        return False, f"expected is not a non-empty string ({type(expected).__name__})"
    exp = expected.strip()
    actuals = actual if isinstance(actual, (list, tuple)) else [actual]
    if not actuals:
        return False, "no actual value"
    if any(not isinstance(a, str) for a in actuals):
        return False, "actual is not a string"

    strippers = [STRIP_DECORATION[r] for r in rules if r in STRIP_DECORATION]
    if strippers:
        for a in actuals:
            if not any(strip(a) == exp for strip in strippers):
                return False, f"differs after removing decoration: {a.strip()[:60]!r}"
        return True, ""

    for a in actuals:
        if exp not in a.strip():
            return False, f"expected text absent from actual: {a.strip()[:60]!r}"
    return True, "excused by containment only (no strip function for this rule)"


def adjudicate(evaluation: dict, rule_names) -> Verdict:
    """Relaxed verdict for one episode from the official evaluation dict.

    `evaluation` is TestTracker.to_dict() -- {"success", "failures": [{"requirement",
    "trace", ...}], ...}. `rule_names` are the rules active for this run (persona rules),
    so a field is only excusable when a rule actually claims it.
    """
    official = bool(evaluation.get("success"))
    params = host_params(rule_names)
    v = Verdict(official=official, relaxed=official)
    failures = evaluation.get("failures") or []
    if official or not failures:
        return v

    for f in failures:
        req = str(f.get("requirement", "")).strip()

        claiming = [r for p, rs in params.items()
                    if re.search(rf"\b{re.escape(p)}\b", req) for r in rs]
        if not claiming:
            v.unexcused.append(req)
            continue
        values = _values_from_trace(str(f.get("trace", "")))
        if values is None:
            v.unexcused.append(req)
            v.notes.append(f"unparsable trace, kept as failure: {req[:60]}")
            continue
        ok, why = _judge(*values, claiming)
        if ok:
            v.excused.append(req)
            if why:
                v.notes.append(f"{why}: {req[:60]}")
        else:
            v.unexcused.append(req)
            v.notes.append(f"{why}: {req[:60]}")

    v.relaxed = not v.unexcused
    return v


def assert_monotonic(verdicts) -> None:
    """Relaxed TGC is a strict relaxation of official, so it may never be lower.

    Cheap invariant worth keeping live: any violation means the adjudicator turned a passing
    episode into a failing one, which it has no mechanism to do legitimately.
    """
    bad = [v for v in verdicts if v.official and not v.relaxed]
    if bad:
        raise AssertionError(
            f"relaxed TGC below official on {len(bad)} episode(s) -- adjudicator bug")
