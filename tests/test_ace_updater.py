"""ACEStyleUpdater: the failure modes that would silently produce a fake plateau.

Q3's headline claim is "self-evolution plateaus above oracle". Several bugs in the updater
produce exactly that curve for the wrong reason, and none of them raises:

  - observing on the WRONG polarity (learning from accepted episodes instead of complaints)
    leaves memory empty forever while still burning one LLM call per episode
  - a JSON parse failure on fenced output silently means "no update this episode"
  - an unbounded memory grows into the regime where Q2 measured length and order effects
    1.89x larger than the thing Q3 is trying to measure

Each is asserted here against MockLLM, so the suite costs no API calls.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from appworld_p.ace_updater import (ACEStyleUpdater, _extract_json_array,  # noqa: E402
                                    embed_text, cosine, parse_bullets, render_bullets)
from appworld_p.feedback import Feedback  # noqa: E402
from appworld_p.llm import MockLLM  # noqa: E402

from conftest import make_episode  # noqa: E402


def complaint(text="sms not signed with first name: 'hi' -- I don't want it done that way."):
    return Feedback(tier="instance", accepted=False, text=f"That's not how I like things done:\n- {text}",
                    provenance=["sms_signoff"])


ACCEPTED = Feedback(tier="instance", accepted=True, text="Thanks, that was done the way I like it.")


def _add(text):
    return f'[{{"action": "add", "text": "{text}", "reason": "test"}}]'


def test_learns_from_complaints_not_from_acceptance():
    """The polarity test. An inverted condition (`not feedback.accepted`) skips every complaint,
    so memory stays empty for the whole sweep and the violation curve is flat by construction."""
    llm = MockLLM([_add("Text messages should be signed with your first name.")])
    up = ACEStyleUpdater(llm)
    up.observe(make_episode([]), complaint())
    assert len(up.bullets) == 1, "a complaint must produce a bullet"
    assert "signed" in up.render_memory()
    assert llm.usage.calls == 1


def test_accepted_episode_costs_nothing():
    llm = MockLLM([_add("should not be written")])
    up = ACEStyleUpdater(llm)
    up.observe(make_episode([]), ACCEPTED)
    assert up.bullets == []
    assert llm.usage.calls == 0, "an accepted episode must not spend an LLM call"


def test_fenced_json_is_parsed():
    """Models wrap JSON in ```json fences unprompted. Treating that as a parse failure turns a
    formatting quirk into a fake plateau."""
    fenced = '```json\n[{"action": "add", "text": "Notes should be lowercase."}]\n```'
    llm = MockLLM([fenced])
    up = ACEStyleUpdater(llm)
    up.observe(make_episode([]), complaint())
    assert len(up.bullets) == 1
    assert up.parse_failures == 0


def test_prose_wrapped_json_is_recovered():
    llm = MockLLM(['Here are the deltas:\n[{"action": "add", "text": "Keep notes short."}]\nDone.'])
    up = ACEStyleUpdater(llm)
    up.observe(make_episode([]), complaint())
    assert len(up.bullets) == 1


def test_unparseable_output_is_counted_not_hidden():
    llm = MockLLM(["I cannot produce JSON for this."])
    up = ACEStyleUpdater(llm)
    up.observe(make_episode([]), complaint())
    assert up.bullets == []
    assert up.parse_failures == 1, "a parse failure must be visible in state(), not silent"
    assert up.state()["parse_failures"] == 1


def test_modify_edits_in_place_and_keeps_id():
    """ACE's stable-id property: a modify must not append a second bullet."""
    llm = MockLLM([_add("Sign texts.")])
    up = ACEStyleUpdater(llm)
    up.observe(make_episode([]), complaint())
    bid = up.bullets[0].id
    up.llm = MockLLM([f'[{{"action": "modify", "id": "{bid}", "text": "Sign every text with Lena."}}]'])
    up.observe(make_episode([]), complaint())
    assert len(up.bullets) == 1, "modify must edit in place, not append"
    assert up.bullets[0].id == bid
    assert up.bullets[0].text == "Sign every text with Lena."


def test_modify_tolerates_bracketed_id_as_rendered():
    """render_bullets prints '- [abc123] text', so the model tends to echo the brackets back."""
    llm = MockLLM([_add("Sign texts.")])
    up = ACEStyleUpdater(llm)
    up.observe(make_episode([]), complaint())
    bid = up.bullets[0].id
    up.llm = MockLLM([f'[{{"action": "modify", "id": "[{bid}]", "text": "Updated."}}]'])
    up.observe(make_episode([]), complaint())
    assert up.bullets[0].text == "Updated."


def test_identical_text_does_not_duplicate():
    line = "Text messages should be signed."
    up = ACEStyleUpdater(MockLLM([_add(line), _add(line)]))
    up.observe(make_episode([]), complaint())
    up.observe(make_episode([]), complaint())
    assert len(up.bullets) == 1


def test_near_duplicates_are_deduped():
    up = ACEStyleUpdater(MockLLM([
        _add("Text messages should be signed with your first name."),
        _add("Text messages must be signed with your first name."),
    ]))
    up.observe(make_episode([]), complaint())
    up.observe(make_episode([]), complaint())
    assert len(up.bullets) == 1, f"near-duplicates survived: {[b.text for b in up.bullets]}"


def test_distinct_preferences_are_kept():
    """Dedup must not be so aggressive that it collapses genuinely different preferences --
    that would cap the learner below the persona's width and manufacture a plateau."""
    up = ACEStyleUpdater(MockLLM([
        _add("Text messages should be signed with your first name."),
        _add("Venmo payments should be marked private."),
    ]))
    up.observe(make_episode([]), complaint())
    up.observe(make_episode([]), complaint())
    assert len(up.bullets) == 2, [b.text for b in up.bullets]


def test_capacity_is_enforced_so_length_stays_constant():
    """The Q2-contamination guard: without a cap the block grows into the length/order regime."""
    cap = 4
    up = ACEStyleUpdater(MockLLM([_add(f"Preference number {i} about topic {i}.") for i in range(10)]),
                         max_bullets=cap)
    for _ in range(10):
        up.observe(make_episode([]), complaint())
    assert len(up.bullets) <= cap
    assert up.state()["memory_lines"] <= cap


def test_state_records_length_and_order():
    up = ACEStyleUpdater(MockLLM([_add("Alpha preference."), _add("Beta preference.")]))
    up.observe(make_episode([]), complaint())
    first = up.state()["order"]
    up.observe(make_episode([]), complaint())
    st = up.state()
    assert st["bullets"] == 2 and st["memory_lines"] == 2
    assert st["memory_chars"] > 0
    assert st["order"] != first and st["order"].startswith(first), \
        "order fingerprint must extend, so a reordering is visible without diffing the text"


def test_render_parse_round_trip():
    up = ACEStyleUpdater(MockLLM([_add("Sign every text message.")]))
    up.observe(make_episode([]), complaint())
    assert [b.text for b in parse_bullets(render_bullets(up.bullets))] == \
           [b.text for b in up.bullets]


def test_empty_memory_renders_a_placeholder_not_a_blank():
    """A blank block and 'no preferences yet' are different prompts; the n=0 checkpoint must be
    the same shape as later ones or the first point is not comparable to the rest."""
    assert render_bullets([]).strip() != ""


def test_extract_json_array_edge_cases():
    assert _extract_json_array('[]') == []
    assert _extract_json_array('{"action": "add", "text": "x"}') == [{"action": "add", "text": "x"}]
    assert _extract_json_array('not json at all') is None
    assert _extract_json_array('') is None


def test_embedding_similarity_is_ordered():
    """The dedup threshold is only meaningful if paraphrases score above unrelated text."""
    a = embed_text("Text messages should be signed with your first name.")
    b = embed_text("Text messages must be signed with your first name.")
    c = embed_text("Venmo payments should be marked private.")
    assert cosine(a, b) > cosine(a, c)
