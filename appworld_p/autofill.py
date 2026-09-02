"""Write-time field autofill: the strongest control-class harness in the study.

The gate arms (`oracle_verifier`, `oracle_verifier_value`) leave the field in f's hands and only
judge the result afterwards. On `spend_total_per_recipient` that is not enough, and the
transcripts say why: the checker already hands back the whole expected sequence
("payment 1 must be $846; payment 2 must be $937"), and the agent still writes $846 into both.
It is not failing to compute the number -- it computed $846 correctly on the first attempt --
it is failing to emit a *different* number per payment within one episode.

So the mechanism that should fix it is not more information but a different action space: the
agent writes the memo it wants, and the harness rewrites the `total: $X` tag at call time from a
ledger it maintains itself. f never has to hold the running sum. This is the same move as the
family-B frequency counter (state lives outside f) applied to a field instead of a choice, and
it is what makes "the reachable set moves when the harness changes" a measurement rather than an
inference: the preference, the actuator, the model, and the context are all held fixed.

Scope note: this repairs only the aggregate the preference asks for. It does not decide whom to
pay, how much, or whether to pay at all -- those stay with f, so a task the agent would have
failed on its own is still failed here.
"""
from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field

TOTAL_TAG = re.compile(r"total:\s*\$?[0-9]+(?:\.[0-9]{1,2})?\s*$", re.I)


@dataclass
class SpendTotalAutofill:
    """Maintains a per-recipient running total and repairs the memo tag on the way through.

    `baseline` maps a recipient key to what the world says was already sent to them before this
    episode -- the same ground truth the rule scores against (driver.read_spend_totals), not a
    re-derivation, so the harness cannot be right for a reason the rule disagrees with.
    """

    baseline: dict[str, float]
    per_recipient: bool = True
    running: dict[str, float] = field(default_factory=dict)
    repairs: list[dict] = field(default_factory=list)

    def _key(self, kwargs: dict) -> str:
        if not self.per_recipient:
            return ""
        for name in ("email", "receiver_email", "recipient_email", "to_email"):
            if kwargs.get(name):
                return str(kwargs[name]).strip().lower()
        return ""

    def apply(self, canonical: str) -> tuple[str, dict | None]:
        """-> (possibly rewritten call, repair record or None).

        Anything that is not a venmo payment passes through untouched. A malformed call is also
        passed through: the actuator's own validator owns that error, and swallowing it here
        would hide a protocol failure behind a preference fix.
        """
        try:
            call = ast.parse(canonical.strip()).body[0].value  # type: ignore[attr-defined]
        except (SyntaxError, AttributeError, IndexError):
            return canonical, None
        if not isinstance(call, ast.Call):
            return canonical, None
        func = call.func
        if not (isinstance(func, ast.Attribute) and isinstance(func.value, ast.Attribute)
                and func.value.attr == "venmo" and func.attr == "create_transaction"):
            return canonical, None

        kwargs = {}
        for kw in call.keywords:
            if kw.arg is None:
                continue
            try:
                kwargs[kw.arg] = ast.literal_eval(kw.value)
            except (ValueError, SyntaxError):
                kwargs[kw.arg] = None
        try:
            amount = float(kwargs.get("amount") or 0)
        except (TypeError, ValueError):
            amount = 0.0

        key = self._key(kwargs)
        self.running[key] = round(self.running.get(key, 0.0) + amount, 2)
        expected = round(self.baseline.get(key, 0.0) + self.running[key], 2)

        note_arg = next((kw for kw in call.keywords
                         if kw.arg in ("description", "note", "memo")), None)
        original = str(kwargs.get(note_arg.arg)) if note_arg is not None else ""
        # strip whatever tag f wrote (right or wrong) and append the ledger's figure. Appending
        # without stripping would leave two tags and the rule reads the last one, which would
        # score as a pass while the memo the recipient sees is still wrong.
        body = TOTAL_TAG.sub("", original).rstrip(" ;,|")
        repaired = f"{body} total: ${expected:g}".strip() if body else f"total: ${expected:g}"
        if note_arg is None:
            call.keywords.append(ast.keyword(arg="description",
                                             value=ast.Constant(value=repaired)))
        else:
            note_arg.value = ast.Constant(value=repaired)
        rewritten = ast.unparse(ast.fix_missing_locations(call))
        record = {"before": original, "after": repaired, "expected": expected,
                  "recipient": key, "amount": amount,
                  "changed": original.strip() != repaired.strip()}
        self.repairs.append(record)
        return rewritten, record
