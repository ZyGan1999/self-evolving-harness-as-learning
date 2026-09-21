"""Fill running-total notes from a ledger of successful Venmo payments.

The proposed amount is included in the note before execution, but is committed
to the ledger only after the API returns a successful transaction.
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
        expected = round(self.baseline.get(key, 0.0) + self.running.get(key, 0.0) + amount, 2)

        source_field = next((name for name in ("description", "note", "memo")
                             if name in kwargs), None)
        original = str(kwargs.get(source_field) or "") if source_field else ""

        body = TOTAL_TAG.sub("", original).rstrip(" ;,|")
        repaired = f"{body} total: ${expected:g}".strip() if body else f"total: ${expected:g}"

        aliases_present = any(kw.arg in ("note", "memo") for kw in call.keywords)
        call.keywords = [kw for kw in call.keywords
                         if kw.arg not in ("description", "note", "memo")]
        call.keywords.append(ast.keyword(arg="description",
                                         value=ast.Constant(value=repaired)))
        rewritten = ast.unparse(ast.fix_missing_locations(call))
        record = {"before": original, "after": repaired, "expected": expected,
                  "recipient": key, "amount": amount,
                  "source_field": source_field,
                  "changed": (original.strip() != repaired.strip()
                              or source_field != "description" or aliases_present)}
        self.repairs.append(record)
        return rewritten, record

    def commit(self, repair: dict, succeeded: bool) -> None:
        """Advance the ledger only after the payment API confirms success."""
        if "succeeded" in repair:
            raise ValueError("Payment outcome was already committed")
        repair["succeeded"] = succeeded
        if succeeded:
            key = repair["recipient"]
            self.running[key] = round(self.running.get(key, 0.0) + repair["amount"], 2)
