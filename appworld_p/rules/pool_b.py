"""History-dependent and computational preference rules."""

import re

from ..episode import EpisodeRecord
from ..history import SessionHistory
from .api_map import find_actions
from .base import Rule, register, sentences, word_count

PRIVATE_NOTE_RE = re.compile(r"^\[(K7|Q3)\|(\d{4}-\d{2}-\d{2})\|([^|\]]{1,60})\]$")


def _amount_cents(call) -> int | None:
    amount = call.arg("amount", "value", "total")
    try:
        return round(float(amount) * 100)
    except (TypeError, ValueError):
        return None


@register
class PrivateNoteFormat(Rule):
    """Check bracketed note format, amount-dependent code, and summary length."""
    name = "private_note_format"
    pool = "A"
    apps = ("venmo",)
    trigger_actions = ("venmo_payment",)
    field_claims = ("venmo.note",)
    oracle_text = ("Venmo payment descriptions must be exactly '[<code>|<date>|<summary>]' where "
                   "<code> is K7 if the whole-dollar part of the amount is an even number and "
                   "Q3 if it is odd, <date> is today's date as YYYY-MM-DD, and <summary> is at "
                   "most 6 words.")
    correction_template = ("Payment notes must look like '[K7|2026-07-07|dinner split]' — "
                           "and the K7/Q3 code depends on the amount.")

    def applicable(self, ep):
        return bool(find_actions(ep, "venmo_payment"))

    def satisfied(self, ep, history):
        for call in find_actions(ep, "venmo_payment"):
            note = str(call.arg("description", "note", "memo", default=""))
            m = PRIVATE_NOTE_RE.match(note.strip())
            if not m:
                return False, f"note not in private format: {note!r}"
            code, _date, summary = m.groups()
            cents = _amount_cents(call)
            if cents is not None:
                expected = "K7" if (cents // 100) % 2 == 0 else "Q3"
                if code != expected:
                    return False, f"derived code wrong: got {code}, expected {expected} (amount)"
            if word_count(summary) > 6:
                return False, f"summary too long: {summary!r}"
        return True, ""


@register
class RecentModeCard(Rule):
    """Sliding-window frequency: assertions go stale; an external counter never does.
    Grounded on venmo card payments (amazon is test_challenge-only in AppWorld)."""
    name = "recent_mode_card"
    pool = "B"
    apps = ("venmo",)
    trigger_actions = ("venmo_payment",)
    field_claims = ("venmo.payment_card",)
    window = 5
    oracle_text = ("When a Venmo payment is made with a payment card, use the card I have "
                   "used most often in my last 5 card payments.")
    correction_template = "Use my usual card — the one I've been using most recently."

    def applicable(self, ep):
        return any(c.arg("payment_card_id", "card_id") is not None
                   for c in find_actions(ep, "venmo_payment"))

    def satisfied(self, ep, history: SessionHistory):
        mode = history.recent_mode_card(self.window)
        if mode is None:
            return True, "no card history yet (vacuously satisfied)"
        for call in find_actions(ep, "venmo_payment"):
            card = call.arg("payment_card_id", "card_id")
            if card is not None and str(card) != mode:
                return False, f"used card {card}, recent-mode card is {mode}"
        return True, ""


@register
class UsualCard(Rule):
    """Family B (statistical aggregation): the user habitually pays with one bank's card.

    The habit is a biased-random distribution pi_u over interchangeable cards; the target
    is argmax pi_u, observable only as a statistic of the interaction stream. Under the
    binary feedback tier (log says which card was used + happy/unhappy, never which card
    was wanted) recovering it requires per-card acceptance-rate credit assignment — which
    f does not do inside a forward pass, but a 20-line external counter does.

    Metric = the FIRST card the episode attempts (intent). Later attempts are confounded:
    a card can fail for insufficient balance and force a fallback.
    """
    name = "usual_card"
    pool = "B"
    apps = ("venmo",)
    trigger_actions = ("venmo_payment",)
    field_claims = ("venmo.payment_card",)

    oracle_text = ("When you send a Venmo payment for me, pay with one of my payment cards "
                   "rather than my Venmo balance, and use the card I habitually use.")
    correction_template = "Use my usual card for Venmo payments — the one I normally pay with."

    def applicable(self, ep):
        calls = find_actions(ep, "venmo_payment")
        if not calls:
            return False

        used_card = any(c.arg("payment_card_id", "card_id") is not None for c in calls)
        return bool(ep.card_names) or not used_card

    def gate_detail(self, ep, history: SessionHistory) -> str:
        """Reject-only channel: name what the attempt used, never what it should have used.

        Retries run in a fresh world with fresh context, so without this the agent cannot tell
        which cards it has already been rejected for and may re-offer the same one -- the gate
        would then measure its luck on attempt 1, not the mechanism. Naming the card the agent
        itself just used adds no information about the target, so the exclusion is admissible
        where `satisfied`'s detail ("habitual card is X") is not. The driver accumulates these
        across attempts, so attempt 3 sees both earlier exclusions.
        """
        for call in find_actions(ep, "venmo_payment"):
            card = call.arg("payment_card_id", "card_id")
            if card is None:
                continue
            bank = ep.card_names.get(str(card))
            return (f"you paid with the {bank} card, which is not the one — pick a different "
                    f"card next time" if bank else
                    f"card id {card} is not one of the user's cards — list the cards first")
        return "you paid from the Venmo balance instead of using a card"

    def satisfied(self, ep, history: SessionHistory):
        target = history.habit_target
        if not target:
            return True, "no habit established yet (vacuously satisfied)"
        for call in find_actions(ep, "venmo_payment"):
            card = call.arg("payment_card_id", "card_id")
            if card is None:
                continue
            bank = ep.card_names.get(str(card))
            if bank is None:
                return False, f"used unknown card id {card} (not one of the user's cards)"
            if bank != target:
                return False, f"first card was {bank}, habitual card is {target}"
            return True, f"first card {bank}"
        return False, "paid without a payment card (balance), habitual card not used"


@register
class TxnRunningCount(Rule):
    """Cross-session exact count: context cannot track it, an external counter can."""
    name = "txn_running_count"
    pool = "B"
    apps = ("venmo",)
    trigger_actions = ("venmo_payment",)
    field_claims = ("venmo.note_suffix",)
    oracle_text = ("Every Venmo payment description must end with ' txn #N' where N is the "
                   "running count of all payments you have ever made for me, this one included.")
    correction_template = "Payment notes must end with the running payment number, like 'txn #17'."

    def applicable(self, ep):
        return bool(find_actions(ep, "venmo_payment"))

    def satisfied(self, ep, history: SessionHistory):
        count = history.payment_count
        for i, call in enumerate(find_actions(ep, "venmo_payment"), start=1):
            note = str(call.arg("description", "note", "memo", default=""))
            m = re.search(r"txn #(\d+)\s*$", note)
            if not m:
                return False, f"note lacks running count: {note!r}"
            if int(m.group(1)) != count + i:
                return False, f"running count wrong: got {m.group(1)}, expected {count + i}"
        return True, ""


@register
class ChecksumSubject(Rule):
    """Derived computation across fields: subject checksum from body word count."""
    name = "checksum_subject"
    pool = "B"
    apps = ("gmail",)
    trigger_actions = ("send_email",)
    field_claims = ("gmail.subject_suffix",)
    oracle_text = ("Email subjects must end with ' #n' where n = (number of words in the "
                   "email body) mod 7.")
    correction_template = "Subjects need the checksum tag: ' #n', n = body word count mod 7."

    def applicable(self, ep):
        return bool(find_actions(ep, "send_email"))

    def satisfied(self, ep, history):
        for call in find_actions(ep, "send_email"):
            subject = str(call.arg("subject", "title", default=""))
            body = str(call.arg("body", "content", "text", default=""))
            m = re.search(r"#(\d)\s*$", subject)
            if not m:
                return False, f"subject lacks checksum tag: {subject!r}"
            if int(m.group(1)) != word_count(body) % 7:
                return False, (f"checksum wrong: got {m.group(1)}, "
                               f"expected {word_count(body) % 7}")
        return True, ""


@register
class SmsChecksum(Rule):
    """Derived computation: sms suffix checksum from its own word count."""
    name = "sms_checksum"
    pool = "B"
    apps = ("phone",)
    trigger_actions = ("send_sms",)
    field_claims = ("phone.message_suffix",)
    oracle_text = ("Text messages you send for me must end with ' #n' where n = (number of "
                   "words in the message, excluding the tag) mod 7.")
    correction_template = "Text messages need the checksum tag at the end: ' #n', n = word count mod 7."

    def applicable(self, ep):
        return bool(find_actions(ep, "send_sms"))

    def satisfied(self, ep, history):
        for call in find_actions(ep, "send_sms"):
            message = str(call.arg("message", "text", default=""))
            m = re.search(r"#(\d)\s*$", message)
            if not m:
                return False, f"sms lacks checksum tag: {message[-40:]!r}"
            stripped = re.sub(r"#\d\s*$", "", message)
            if int(m.group(1)) != word_count(stripped) % 7:
                return False, (f"checksum wrong: got {m.group(1)}, "
                               f"expected {word_count(stripped) % 7}")
        return True, ""


@register
class SmsCharChecksum(Rule):
    """Check the message-body character count modulo seven."""
    name = "sms_char_checksum"
    pool = "B"
    apps = ("phone",)
    trigger_actions = ("send_sms",)
    field_claims = ("phone.message_suffix",)
    oracle_text = ("Text messages you send for me must end with ' #n' where n = (the total "
                   "number of characters in the message text, counting spaces and "
                   "punctuation but excluding the ' #n' tag itself) mod 7.")
    correction_template = ("Text messages need the character-count checksum tag at the "
                           "end: ' #n', n = number of characters mod 7.")

    def applicable(self, ep):
        return bool(find_actions(ep, "send_sms"))

    def satisfied(self, ep, history):
        for call in find_actions(ep, "send_sms"):
            message = str(call.arg("message", "text", default=""))
            m = re.search(r"#(\d)\s*$", message)
            if not m:
                return False, f"sms lacks checksum tag: {message[-40:]!r}"
            stripped = re.sub(r"\s*#\d\s*$", "", message)
            expected = len(stripped) % 7
            if int(m.group(1)) != expected:
                return False, (f"checksum wrong: got {m.group(1)}, expected {expected} "
                               f"({len(stripped)} chars)")
        return True, ""


@register
class NoteExactWords(Rule):
    """Generation-hard, verification-easy: exact word count."""
    name = "note_exact_words"
    pool = "B"
    apps = ("simple_note",)
    trigger_actions = ("create_note",)
    field_claims = ("simple_note.content",)
    target_words = 25
    oracle_text = "Note bodies you create must be exactly 25 words long."
    correction_template = "Note bodies must be exactly 25 words — count them."

    def applicable(self, ep):
        return bool(find_actions(ep, "create_note"))

    def satisfied(self, ep, history):
        for call in find_actions(ep, "create_note"):
            content = str(call.arg("content", "body", "text", default=""))
            if word_count(content) != self.target_words:
                return False, f"note body has {word_count(content)} words, want {self.target_words}"
        return True, ""


@register
class LipogramEmailClose(Rule):
    """Generation-hard, verification-easy: final body line avoids the letter 'e'."""
    name = "lipogram_email_close"
    pool = "B"
    apps = ("gmail",)
    trigger_actions = ("send_email",)
    field_claims = ("gmail.body",)
    oracle_text = ("The last line of every email body must be a short closing sentence "
                   "that does not contain the letter 'e' (case-insensitive).")
    correction_template = "Close emails with a final line containing no letter 'e'."

    def applicable(self, ep):
        return bool(find_actions(ep, "send_email"))

    def satisfied(self, ep, history):
        for call in find_actions(ep, "send_email"):
            body = str(call.arg("body", "content", "text", default="")).strip()
            if not body:
                return False, "empty email body"
            last_line = body.splitlines()[-1]
            if "e" in last_line.lower():
                return False, f"closing line contains 'e': {last_line!r}"
            if not sentences(last_line):
                return False, "no closing sentence"
        return True, ""


class _SpendRunningTotal(Rule):
    """Check creation-time notes against totals of successful outgoing payments."""
    pool = "B"
    apps = ("venmo",)
    trigger_actions = ("venmo_payment",)
    field_claims = ("venmo.note_suffix",)

    def _payments(self, ep: EpisodeRecord):
        payments = find_actions(ep, "venmo_payment")
        if any(c.succeeded is None for c in payments):
            raise ValueError("Running-total scoring requires recorded payment outcomes")
        return [c for c in payments if c.succeeded]

    def applicable(self, ep: EpisodeRecord) -> bool:
        if not self._payments(ep):
            return False
        if not ep.spend_totals:
            raise ValueError("Successful payments require an initial ledger for scoring")
        return True

    def _baseline(self, ep: EpisodeRecord, call) -> float | None:
        raise NotImplementedError

    def satisfied(self, ep: EpisodeRecord, history: SessionHistory) -> tuple[bool, str]:
        """Accumulate successful payments and report all incorrect note totals."""
        running: dict[str, float] = {}
        problems: list[str] = []
        for i, call in enumerate(self._payments(ep), 1):
            note = str(call.arg("description", default=""))
            m = re.search(r"total:\s*\$?([0-9]+(?:\.[0-9]{1,2})?)\s*$", note, re.I)
            base = self._baseline(ep, call)
            if base is None:
                problems.append(f"payment {i}: cannot resolve the recipient")
                continue
            try:
                amount = float(call.arg("amount", default=0) or 0)
            except (TypeError, ValueError):
                amount = 0.0
            key = self._key(ep, call) or ""

            running[key] = round(running.get(key, 0.0) + amount, 2)
            expected = round(base + running[key], 2)

            if not m:
                problems.append(f"payment {i} (${amount:g}): note {note!r} lacks the "
                                f"'total: $X' tag; it must end with 'total: ${expected:g}'")
                continue
            got = float(m.group(1))
            if abs(got - expected) > 0.005:
                problems.append(f"payment {i} (${amount:g}): note says total ${got:g}, "
                                f"must be ${expected:g} (${base:g} already sent to this "
                                f"person + ${running[key]:g} this episode)")
        if problems:
            return False, "; ".join(problems)
        return True, ""


@register
class SpendTotalPerRecipient(_SpendRunningTotal):
    """Easier rung: sum only what was already sent to THIS recipient (7-14 payments)."""
    name = "spend_total_per_recipient"
    oracle_text = ("When you send someone money on Venmo for me, end the payment note with "
                   "'total: $X', where X is the running total I have ever sent that same "
                   "person on Venmo, this payment included. Use two decimal places only if "
                   "the amount has cents.")
    correction_template = ("Payment notes must end with the running total sent to that "
                           "person, like 'total: $415'.")

    def _key(self, ep: EpisodeRecord, call) -> str | None:
        return call.arg("receiver_email", "email", "receiver", default=None)

    def _baseline(self, ep: EpisodeRecord, call) -> float | None:
        who = self._key(ep, call)
        if who is None:
            return None
        return float(ep.spend_totals.get("per_recipient", {}).get(str(who), 0.0))


@register
class SpendTotalAllTime(_SpendRunningTotal):
    """Check the cumulative amount of all successful outgoing payments."""
    name = "spend_total_all_time"
    oracle_text = ("When you send money on Venmo for me, end the payment note with "
                   "'total: $X', where X is the running total of every Venmo payment I have "
                   "ever sent to anyone, this payment included. Use two decimal places only "
                   "if the amount has cents.")
    correction_template = ("Payment notes must end with my all-time Venmo spending total, "
                           "like 'total: $5230'.")

    def _key(self, ep: EpisodeRecord, call) -> str | None:
        return "__all__"

    def _baseline(self, ep: EpisodeRecord, call) -> float | None:
        return float(ep.spend_totals.get("total", 0.0))
