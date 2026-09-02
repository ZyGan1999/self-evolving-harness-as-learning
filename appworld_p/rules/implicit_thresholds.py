"""Implicit-threshold rules: the checker tests a bound the complaint never states.

Every other pool-A rule is a form the learner either reproduces or does not -- a greeting, a
signoff, a lowercase note. Under precise feedback haiku learned all of them, driving violations
from 0.745 to 0.062 over 48 episodes with no ceiling, so a rule whose target is a POINT ON A
NUMBER LINE is the next thing to try: partial success is available in a way it is not for a form.

The learner sees "payment note too long: 'Refund for accidental payment request'" and must infer
the bound. Nothing in the complaint says "three". If it settles on four or five words as good
enough, that is a plateau -- and unlike the weak-feedback arms it is a plateau reached by
optimising, because the feedback stays at the instance tier and names the artefact precisely.

The oracle arm states the number outright ("three words or fewer"), so oracle is reachable and
the gap between the two is optimisation error rather than Q1's approximation error.

Bounds are chosen from the observed data, not guessed. Task-dictated payment notes in the pool run
2-6 words ("Refund for accidental payment request" is 5, "Groceries" is 1), and message bodies run
4-55 words, so 3 and 8 both sit inside the range the agent naturally produces: violations are
frequent enough to give the learner signal, and compliance is possible without fighting the task
instruction. A bound outside that range would make the rule either vacuous or impossible, which is
how sms_terse and playlist_over_like became unusable (docs/Q3.md 3.7).
"""

from ..episode import EpisodeRecord
from ..history import SessionHistory
from .api_map import find_actions
from .base import Rule, register

# Both bounds are set from the MEASURED text in this task pool, not from the prose in the
# instructions, because the two disagree. The 29caf6f movie lists read as 38-55 characters of
# comma-separated titles but only 7 whitespace tokens ('Network, The Godfather, Django Unchained,
# Heat, Casino'), and the 0d8a4ee dictated bodies are 4-6 tokens. So the whole observed range is
# 4-7 words:
#
#   'Please get on venmo.'                                   4
#   'Get on venmo please!'                                   4
#   'Make an account on venmo please.'                       6
#   'Hi Sam the package arrived Lena'                        6
#   'Network, The Godfather, Django Unchained, Heat, Casino'  7
#
# A bound of 8 sits ABOVE all of it and would make the rule vacuous -- base ~= 0, the same defect
# that removed payment_note_no_brackets and playlist_private (docs/Q3.md 3.7). 5 splits the range:
# the two 4-word bodies comply, the 6- and 7-word ones do not, so violations are frequent without
# being forced. Notes run 1-5 words, so 3 splits theirs the same way.
NOTE_MAX_WORDS = 3
SMS_MAX_WORDS = 5


def _words(text: str) -> int:
    return len(text.split())


@register
class PaymentNoteBrief(Rule):
    """Venmo notes must be at most NOTE_MAX_WORDS; the complaint says only 'too long'."""

    name = "payment_note_brief"
    pool = "A"
    apps = ("venmo",)
    trigger_actions = ("venmo_payment",)
    # note_length, so this cannot be composed with payment_note_single_word (which claims the
    # same field for a different bound) -- build_rules raises on the collision.
    field_claims = ("venmo.note_length",)
    oracle_text = ("Venmo payment descriptions must be at most three words long.")
    negation_text = ("Venmo payment descriptions should be full sentences that explain the "
                     "payment in detail.")
    correction_template = "Keep payment descriptions to three words or fewer."

    def applicable(self, ep: EpisodeRecord) -> bool:
        return bool(find_actions(ep, "venmo_payment"))

    def satisfied(self, ep: EpisodeRecord, history: SessionHistory) -> tuple[bool, str]:
        for call in find_actions(ep, "venmo_payment"):
            note = str(call.arg("description", "note", "memo", default="")).strip()
            if _words(note) > NOTE_MAX_WORDS:
                # The count of the OFFENDING note is shown, never the bound. That is the whole
                # design: the learner gets an upper witness ("5 was too many") and must find the
                # threshold by narrowing, which is where a stopping point can appear.
                return False, f"payment note too long ({_words(note)} words): {note[:48]!r}"
        return True, ""

    def gate_detail(self, ep: EpisodeRecord, history: SessionHistory) -> str:
        """Reject-only channel: restate that it was too long without naming the bound."""
        for call in find_actions(ep, "venmo_payment"):
            note = str(call.arg("description", "note", "memo", default="")).strip()
            if _words(note) > NOTE_MAX_WORDS:
                return f"the payment note was too long at {_words(note)} words"
        return ""


@register
class SmsNotTerse(Rule):
    """Text messages must be at most SMS_MAX_WORDS; the complaint says only 'too long'.

    Shares its bound with sms_terse but not its fate. sms_terse is unusable in a persona that also
    carries sms_greeting and sms_signoff, because greeting + dictated body + signoff exceeds five
    words by construction (docs/Q3.md 3.5) -- p11 carries neither, so the bound is reachable: the
    agent can send the dictated body alone and comply on the 4-word tasks.

    What makes this rule different from sms_terse is the CHANNEL, not the bound. sms_terse's
    complaint states the number ('message longer than five words: 66'); this one reports only the
    offending count, so the learner has an upper witness and no threshold.
    """

    name = "sms_not_terse"
    pool = "A"
    apps = ("phone",)
    trigger_actions = ("send_sms",)
    field_claims = ("phone.message_length",)
    oracle_text = "Text messages you send for me must be at most five words long."
    negation_text = ("Text messages you send for me should be detailed and give the full "
                     "context.")
    correction_template = "Keep text messages to five words or fewer."

    def applicable(self, ep: EpisodeRecord) -> bool:
        return bool(find_actions(ep, "send_sms"))

    def satisfied(self, ep: EpisodeRecord, history: SessionHistory) -> tuple[bool, str]:
        for call in find_actions(ep, "send_sms"):
            message = str(call.arg("message", "text", default="")).strip()
            if _words(message) > SMS_MAX_WORDS:
                return False, (f"text message too long ({_words(message)} words): "
                               f"{message[:48]!r}")
        return True, ""

    def gate_detail(self, ep: EpisodeRecord, history: SessionHistory) -> str:
        for call in find_actions(ep, "send_sms"):
            message = str(call.arg("message", "text", default="")).strip()
            if _words(message) > SMS_MAX_WORDS:
                return f"the text message was too long at {_words(message)} words"
        return ""
