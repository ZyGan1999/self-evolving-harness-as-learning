"""Unit tests for the feedback -> updater -> memory loop (no AppWorld needed)."""

from appworld_p.feedback import make_feedback
from appworld_p.rules import RULE_REGISTRY, RuleResult
from appworld_p.updaters import (AssertionTopL, ThresholdWriter, UnbiasedStructured,
                                 build_candidate_bank)
from conftest import call, make_episode


def _results(violated: list[str], satisfied: list[str]) -> list[RuleResult]:
    return ([RuleResult(n, True, False, "bad") for n in violated]
            + [RuleResult(n, True, True) for n in satisfied])


TEMPLATES = {n: RULE_REGISTRY[n].correction_template for n in RULE_REGISTRY}


def test_feedback_tiers():
    results = _results(["todoist_due_date"], ["email_subject_short"])
    default = make_feedback(results, "default")
    assert not default.accepted and "wasn't happy" in default.text
    corrective = make_feedback(results, "corrective", TEMPLATES)
    assert TEMPLATES["todoist_due_date"] in corrective.text
    assert TEMPLATES["email_subject_short"] not in corrective.text
    bypass = make_feedback(results, "bypass")
    assert bypass.structured == {"todoist_due_date": False, "email_subject_short": True}
    accepted = make_feedback(_results([], ["todoist_due_date"]), "corrective", TEMPLATES)
    assert accepted.accepted


def test_assertion_top_l_learns_from_corrections():
    bank = build_candidate_bank(["todoist_due_date", "email_subject_short"],
                                ["file_kebab_case"])
    updater = AssertionTopL(bank, L=1)
    ep = make_episode([call("todoist", "create_task", title="x")])
    for _ in range(3):
        fb = make_feedback(_results(["todoist_due_date"], []), "corrective", TEMPLATES)
        updater.observe(ep, fb)
    memory = updater.render_memory()
    assert RULE_REGISTRY["todoist_due_date"].oracle_text in memory
    assert RULE_REGISTRY["file_kebab_case"].oracle_text not in memory


def test_top_l_pads_with_low_confidence_when_l_large():
    """L > evidence-supported candidates forces low-confidence assertions into memory
    (the mechanism behind Theorem 2's rising branch)."""
    bank = build_candidate_bank(["todoist_due_date"], ["file_kebab_case", "note_title_dated"])
    updater = AssertionTopL(bank, L=3)
    ep = make_episode([call("todoist", "create_task", title="x")])
    updater.observe(ep, make_feedback(_results(["todoist_due_date"], []),
                                      "corrective", TEMPLATES))
    memory = updater.render_memory()
    assert memory.count("- ") == 3  # all three written, two with zero evidence


def test_threshold_writer_restrains():
    bank = build_candidate_bank(["todoist_due_date"], ["file_kebab_case", "note_title_dated"])
    updater = ThresholdWriter(bank, p_bar=0.5)
    ep = make_episode([call("todoist", "create_task", title="x")])
    assert updater.render_memory() == ""  # nothing confident yet
    for _ in range(4):
        updater.observe(ep, make_feedback(_results(["todoist_due_date"], []),
                                          "corrective", TEMPLATES))
    memory = updater.render_memory()
    assert RULE_REGISTRY["todoist_due_date"].oracle_text in memory
    assert memory.count("- ") == 1  # distractors stay out


def test_unbiased_structured_uses_bypass_only():
    bank = build_candidate_bank(["todoist_due_date"], [])
    updater = UnbiasedStructured(bank)
    ep = make_episode([call("todoist", "create_task", title="x")])
    updater.observe(ep, make_feedback(_results(["todoist_due_date"], []), "bypass"))
    assert RULE_REGISTRY["todoist_due_date"].oracle_text in updater.render_memory()
    # corrective-tier feedback carries no structured verdicts -> no learning
    updater2 = UnbiasedStructured(bank)
    updater2.observe(ep, make_feedback(_results(["todoist_due_date"], []),
                                       "corrective", TEMPLATES))
    assert updater2.render_memory() == ""
