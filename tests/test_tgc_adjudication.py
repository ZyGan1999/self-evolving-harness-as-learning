"""Relaxed TGC must excuse decoration and nothing else.

The failure mode that matters is over-excusing: relaxed TGC is the co-metric backing "the
preference was followed and the task still got done", so anything it waves through weakens
that claim. Most cases below therefore assert a failure is KEPT.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from appworld_p.tgc import Verdict, adjudicate, assert_monotonic, host_params  # noqa: E402

Q1_RULES = ["private_note_format", "sms_char_checksum"]


def failure(requirement: str, actual: str, expected: str) -> dict:
    """A failure dict shaped like AppWorld's, including the trace layout we parse."""
    return {
        "requirement": requirement,
        "label": "no_op_fail",
        "trace": ("```python\nwith test(...):\n    test.case(...)\n```\n----------\n"
                  f"AssertionError:  {actual!r} == {expected!r}\n\n"
                  f"Original values:\n{actual!r} == {expected!r}\n"),
    }


def evaluation(*failures, success=False) -> dict:
    return {"success": success, "num_tests": 10, "passes": [], "failures": list(failures)}


SMS_REQ = "assert the added global_text_message has message, public_data.message_text."
NOTE_REQ = "assert the added transaction has description public_data.note."
AMOUNT_REQ = "assert the added transaction has amount private_data.grocery_cost."


def test_pure_suffix_decoration_is_excused():
    ev = evaluation(failure(SMS_REQ, "Done. #5", "Done."))
    v = adjudicate(ev, Q1_RULES)
    assert v.relaxed is True and v.official is False
    assert v.excused == [SMS_REQ] and v.unexcused == []


def test_wrapper_decoration_is_excused():
    """private_note_format wraps the original note; the original survives verbatim."""
    ev = evaluation(failure(NOTE_REQ, "[K7|2023-05-18|Grocery Bill]", "Grocery Bill"))
    assert adjudicate(ev, Q1_RULES).relaxed is True


def test_extra_content_beyond_decoration_is_kept():
    """The whole point of containment over a blanket pass: added prose is still a failure."""
    ev = evaluation(failure(SMS_REQ, "Done. Also I hate you. #5", "Done."))
    v = adjudicate(ev, Q1_RULES)
    assert v.relaxed is False
    assert v.unexcused == [SMS_REQ]


def test_wrong_content_is_kept():
    ev = evaluation(failure(SMS_REQ, "Sent the thing. #5", "Done."))
    assert adjudicate(ev, Q1_RULES).relaxed is False


def test_non_preference_field_is_kept():
    """An amount mismatch is never decoration, even alongside an excusable failure."""
    ev = evaluation(failure(SMS_REQ, "Done. #5", "Done."),
                    failure(AMOUNT_REQ, "12.0", "45.0"))
    v = adjudicate(ev, Q1_RULES)
    assert v.relaxed is False
    assert v.excused == [SMS_REQ] and v.unexcused == [AMOUNT_REQ]


def test_numeric_comparison_on_a_claimed_param_is_kept():
    """Even if a requirement mentions a claimed token, non-string operands never excuse."""
    ev = evaluation({"requirement": "assert the added transaction has description count.",
                     "label": "no_op_fail",
                     "trace": "AssertionError:  3 == 1\n"})
    assert adjudicate(ev, Q1_RULES).relaxed is False


def test_unparsable_trace_is_kept():
    """Silence beats a guess: a trace we cannot read must not be excused."""
    ev = evaluation({"requirement": SMS_REQ, "label": "no_op_fail",
                     "trace": "AssertionError: something went sideways\n"})
    v = adjudicate(ev, Q1_RULES)
    assert v.relaxed is False and v.notes


def test_all_equals_list_form_requires_every_element_decorated():
    """'all ==' puts a list on the left; one undecorated element keeps the failure."""
    ok = evaluation(failure(SMS_REQ, "x", "x"))
    ok["failures"][0]["trace"] = "AssertionError:  ['Done. #5', 'Done. #5'] == 'Done.'\n"
    assert adjudicate(ok, Q1_RULES).relaxed is True

    bad = evaluation(failure(SMS_REQ, "x", "x"))
    bad["failures"][0]["trace"] = "AssertionError:  ['Done. #5', 'Nope.'] == 'Done.'\n"
    assert adjudicate(bad, Q1_RULES).relaxed is False


def test_multiline_assertion_layout_is_parsed():
    """AppWorld breaks long comparisons across lines; missing that left real cases unexcused."""
    ev = evaluation(failure(SMS_REQ, "x", "x"))
    ev["failures"][0]["trace"] = (
        "AssertionError:\n"
        "'make an account on venmo please. #5'\n"
        "==\n"
        "'make an account on venmo please.'\n\n"
        "Original values:\n"
        "'Make an account on venmo please. #5'\n==\n'Make an account on venmo please.'\n")
    v = adjudicate(ev, Q1_RULES)
    assert v.relaxed is True and v.notes == []


def test_multiline_layout_still_rejects_extra_content():
    ev = evaluation(failure(SMS_REQ, "x", "x"))
    ev["failures"][0]["trace"] = (
        "AssertionError:\n'please. and more stuff #5'\n==\n'please.'\n")
    assert adjudicate(ev, Q1_RULES).relaxed is False


def test_rule_not_active_does_not_excuse_its_field():
    """Only rules the persona actually states can excuse their host field."""
    ev = evaluation(failure(SMS_REQ, "Done. #5", "Done."))
    assert adjudicate(ev, ["private_note_format"]).relaxed is False


def test_official_pass_stays_pass_and_is_untouched():
    v = adjudicate(evaluation(success=True), Q1_RULES)
    assert v.official is True and v.relaxed is True and v.excused == []


def test_empty_expected_is_not_excused():
    """Everything contains the empty string; excusing that would pass anything."""
    ev = evaluation(failure(SMS_REQ, "Done. #5", ""))
    assert adjudicate(ev, Q1_RULES).relaxed is False


def test_non_additive_rules_contribute_no_excusable_fields():
    """email_subject_short shortens its field, so containment must not apply to it."""
    assert host_params(["email_subject_short"]) == {}
    assert host_params(["sms_char_checksum"]) == {"message": ["sms_char_checksum"]}


def test_monotonicity_invariant_flags_impossible_verdicts():
    assert_monotonic([Verdict(official=True, relaxed=True),
                      Verdict(official=False, relaxed=True)])
    with pytest.raises(AssertionError, match="below official"):
        assert_monotonic([Verdict(official=True, relaxed=False)])


def test_spend_total_stripper_is_anchored():
    """The running-total suffix is decoration; a note that merely names a price is not."""
    from appworld_p.tgc import STRIP_DECORATION

    strip = STRIP_DECORATION["spend_total_per_recipient"]
    assert strip("Dinner total: $1178") == "Dinner"
    assert strip("Dinner, total: $1178.50") == "Dinner"
    assert strip("Rent") == "Rent"
    # no 'total:' keyword -> leave it alone, or a real task failure would be excused
    assert strip("Paid $50 for gas") == "Paid $50 for gas"
    assert STRIP_DECORATION["spend_total_all_time"] is strip
