"""The baseline recipes must differ from ACE in the ways their papers claim.

A baseline that silently behaves like ACE measures nothing, and the failure is invisible from the
violation curve alone -- both arms would just plateau. So each property the 2x2 in
baseline_updaters.py depends on is asserted here against a scripted LLM, not against a live one:
Reflexion evicts and never edits, Dynamic Cheatsheet regenerates and is unbounded, A-MEM never
evicts and rewrites only a neighbour's framing.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from appworld_p.apicalls import ApiCall  # noqa: E402
from appworld_p.baseline_updaters import (AMemUpdater, DynamicCheatsheetUpdater,  # noqa: E402
                                          ReflexionUpdater, TepaUpdater, TraceUpdater)
from appworld_p.episode import EpisodeRecord  # noqa: E402
from appworld_p.feedback import Feedback  # noqa: E402
from appworld_p.llm import BaseLLM  # noqa: E402


class ScriptedLLM(BaseLLM):
    """Returns queued replies in order, so a recipe's bookkeeping is tested without sampling."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def generate(self, system, messages, max_tokens=512, temperature=0.7, **kw):
        self.calls.append(messages[-1]["content"])
        return self.replies.pop(0) if self.replies else ""


def _ep():
    return EpisodeRecord(task_id="t", instruction="do the thing", supervisor={},
                         api_calls=[], task_completed=True, phase="train", session_index=1)


def _fb(text="that is not how I like it"):
    return Feedback("instance", False, text, provenance=["r"])


def _ep_with_call(app, api, arguments):
    """An episode carrying one API call, for exercising TRACE's compiled predicates."""
    return EpisodeRecord(task_id="t", instruction="do the thing", supervisor={},
                         api_calls=[ApiCall(app=app, api=api, arguments=arguments)],
                         task_completed=True, phase="eval", session_index=-1)


def test_reflexion_window_evicts_and_never_edits():
    """W=2, three complaints: the buffer holds the last two and the first is gone.

    This is the arm's whole diagnostic value -- a FIFO window cannot hold more preferences than
    its width, so if eviction silently did not happen the arm would duplicate ACE's cell.
    """
    llm = ScriptedLLM(["lesson one", "lesson two", "lesson three"])
    u = ReflexionUpdater(llm, window=2)
    for _ in range(3):
        u.observe(_ep(), _fb())
    mem = u.render_memory()
    assert "lesson one" not in mem, "oldest reflection should have fallen out of the window"
    assert "lesson two" in mem and "lesson three" in mem
    assert u.state()["dropped_by_window"] == 1
    # append-only: the surviving text is verbatim what the LLM produced, never rewritten
    assert mem == "- lesson two\n- lesson three"


def test_reflexion_ignores_accepted_episodes():
    llm = ScriptedLLM(["should not be used"])
    u = ReflexionUpdater(llm, window=3)
    u.observe(_ep(), Feedback("instance", True, "thanks"))
    assert u.render_memory() == "" and u.state()["updates"] == 0
    assert llm.calls == [], "an accepted episode must not cost an LLM call"


def test_cheatsheet_regenerates_wholesale_and_is_unbounded():
    """The curator's output REPLACES the sheet, and nothing caps its growth.

    Both halves matter: replacement is what separates it from ACE's targeted edit, and the
    absence of a cap is what separates it from Reflexion's window.
    """
    llm = ScriptedLLM(["- a\n- b", "- a\n- b\n- c\n- d\n- e"])
    u = DynamicCheatsheetUpdater(llm, max_memory_chars=4000)
    u.observe(_ep(), _fb())
    assert u.render_memory() == "- a\n- b"
    u.observe(_ep(), _fb())
    # wholesale replacement, and the sheet grew: no fixed entry budget
    assert u.render_memory() == "- a\n- b\n- c\n- d\n- e"
    assert u.state()["memory_lines"] == 5
    assert u.history[-1]["chars_after"] > u.history[-1]["chars_before"]


def test_amem_never_evicts():
    """Unbounded growth is the cell this arm exists to occupy."""
    llm = ScriptedLLM([
        '{"content": "note one", "keywords": ["a"], "context": "ctx one"}',
        '{"content": "totally unrelated subject matter", "keywords": ["z"], "context": "ctx two"}',
        '{"content": "a third distinct lesson entirely", "keywords": ["q"], "context": "ctx three"}',
    ])
    u = AMemUpdater(llm, top_k=3, link_threshold=0.99)   # threshold high: isolate growth from evolution
    for _ in range(3):
        u.observe(_ep(), _fb())
    assert u.state()["notes"] == 3, "A-MEM must not evict; it has no capacity bound"
    mem = u.render_memory()
    for frag in ("note one", "unrelated subject matter", "third distinct lesson"):
        assert frag in mem


def test_amem_evolution_rewrites_neighbour_framing_not_content():
    """The neighbour-update step may change context/keywords and must leave content intact.

    If content could change, this arm would be a delta-rewrite of the answer-bearing text and
    would stop being distinguishable from ACE's modify action.
    """
    llm = ScriptedLLM([
        '{"content": "sign messages with the first name", "keywords": ["sms"], "context": "old ctx"}',
        # second note is near-identical, so it links and triggers the evolve call
        '{"content": "sign messages with the first name always", "keywords": ["sms"], "context": "new ctx"}',
        '{"update": true, "context": "REVISED CONTEXT", "keywords": ["sms", "signoff"]}',
    ])
    u = AMemUpdater(llm, top_k=3, link_threshold=0.3)
    u.observe(_ep(), _fb())
    first_content = u.notes[0].content
    u.observe(_ep(), _fb())
    assert u.state()["evolved"] == 1, "a near-duplicate note should have triggered evolution"
    assert u.notes[0].content == first_content, "content is immutable under evolution"
    assert u.notes[0].context == "REVISED CONTEXT"
    assert u.notes[0].links, "linked notes must record the link"


def test_amem_parse_failure_is_counted_not_swallowed():
    """A malformed note must be visible in the audit, or a plateau caused by a formatting quirk
    reads as a finding about self-evolution."""
    u = AMemUpdater(ScriptedLLM(["not json at all"]))
    u.observe(_ep(), _fb())
    assert u.state()["parse_failures"] == 1 and u.state()["notes"] == 0


# --------------------------------------------------------------------- TEPA (arXiv:2608.07429)

def test_tepa_revokes_within_key_and_keeps_audit():
    """Contradiction under the SAME key revokes the old precedent but must not delete it.

    Revocation-with-audit is the mechanism; if the old precedent vanished, this arm would be a
    last-write-wins cache, which is precisely the baseline the paper distinguishes itself from.
    """
    llm = ScriptedLLM([
        '{"key": "sms.ending", "content": "sign texts with my first name"}',
        '{"key": "sms.ending", "content": "do not sign texts at all"}',
    ])
    u = TepaUpdater(llm)
    u.observe(_ep(), _fb())
    u.observe(_ep(), _fb())
    st = u.state()
    assert st["revocations"] == 1
    assert st["active"] == 1, "only the newer precedent should be active"
    assert st["stored"] == 2, "the revoked precedent must remain in the store for audit"
    mem = u.render_memory()
    assert "do not sign texts" in mem
    assert "first name" not in mem, "a revoked precedent must leave the injected memory"


def test_tepa_replaces_even_a_mere_restatement():
    """Replacement is unconditional within a key, not gated on contradiction.

    This is the regression the first real run exposed: with an LLM contradiction gate, a writer
    that RESTATES a preference ("must include a category tag" then "must include a category tag
    (e.g. 'Groceries')") never triggers revocation, the active set grows without bound, and the arm
    silently becomes the append-only baseline TEPA's own paper reports falling below no-memory.
    """
    llm = ScriptedLLM([
        '{"key": "payment_note.prefix", "content": "notes must include a category tag"}',
        '{"key": "payment_note.prefix", "content": "notes must include a category tag (e.g. Groceries)"}',
    ])
    u = TepaUpdater(llm)
    u.observe(_ep(), _fb())
    u.observe(_ep(), _fb())
    st = u.state()
    assert st["revocations"] == 1, "a restatement under the same key must still replace"
    assert st["active"] == 1, "the active set is bounded by distinct keys, not by contradiction"


def test_tepa_snaps_a_drifting_key_onto_the_vocabulary():
    """An off-vocabulary key must snap, or replacement never fires for that preference again.

    On the first real run the writer spread ONE preference across payment_note.category_tag,
    .prefix, .format and .suffix, so the active set reached 23 lines over 9 keys with zero
    revocations.
    """
    llm = ScriptedLLM([
        '{"key": "payment_note.prefix", "content": "start notes with a category tag"}',
        '{"key": "payment_note.prefix_tag", "content": "start notes with the category tag"}',
    ])
    u = TepaUpdater(llm)
    u.observe(_ep(), _fb())
    u.observe(_ep(), _fb())
    st = u.state()
    assert st["key_snaps"] == 1, "'payment_note.prefix_tag' should snap to 'payment_note.prefix'"
    assert st["active"] == 1 and st["revocations"] == 1


def test_tepa_does_not_revoke_across_keys():
    """A preference about a different attribute must survive: revocation is within-key only."""
    llm = ScriptedLLM([
        '{"key": "sms.ending", "content": "sign texts with my first name"}',
        '{"key": "payment.visibility", "content": "keep transfers private"}',
    ])
    u = TepaUpdater(llm)
    u.observe(_ep(), _fb())
    u.observe(_ep(), _fb())
    # no conflict prompt should have been consumed, since the keys differ
    assert u.state()["revocations"] == 0 and u.state()["active"] == 2
    assert u.state()["keys"] == 2


def test_tepa_repromotes_revoked_evidence_on_reappearance():
    """Identical evidence returning means the revocation was premature; the paper keeps revoked
    history so it can be re-promoted rather than re-added as a duplicate."""
    llm = ScriptedLLM([
        '{"key": "sms.ending", "content": "sign texts with my first name"}',
        '{"key": "sms.ending", "content": "do not sign texts at all"}',
        '{"key": "sms.ending", "content": "sign texts with my first name"}',
    ])
    u = TepaUpdater(llm)
    for _ in range(3):
        u.observe(_ep(), _fb())
    st = u.state()
    assert st["repromotions"] == 1
    assert st["stored"] == 2, "re-promotion must not create a third precedent"
    assert "first name" in u.render_memory()


# -------------------------------------------------------------------- TRACE (arXiv:2606.13174)

def test_trace_compiles_a_check_that_executes():
    """The compiled predicate must actually reject a violating episode and pass a clean one.

    This is the property that makes TRACE different from every other arm: the memory is not only
    stated, it is enforced.
    """
    llm = ScriptedLLM(['{"api": "phone.send_text_message", "argument": "message",'
                       ' "predicate": "endswith", "value": "Lena",'
                       ' "rule": "Sign text messages with my first name."}'])
    u = TraceUpdater(llm)
    u.observe(_ep(), _fb())
    assert u.state()["rules"] == 1
    assert "Sign text messages" in u.render_memory()

    bad = EpisodeRecord(task_id="t", instruction="i", supervisor={},
                        api_calls=[ApiCall(app="phone", api="send_text_message",
                                           arguments={"message": "package arrived"})],
                        task_completed=True, phase="eval")
    good = EpisodeRecord(task_id="t", instruction="i", supervisor={},
                         api_calls=[ApiCall(app="phone", api="send_text_message",
                                            arguments={"message": "package arrived Lena"})],
                         task_completed=True, phase="eval")
    assert u.check(bad), "the compiled check should reject the unsigned message"
    assert not u.check(good), "the compiled check should pass the signed message"


def test_trace_rule_is_inapplicable_when_its_api_is_absent():
    """A rule whose api never fires must neither pass nor fail, matching the persona checkers'
    applicable/satisfied convention. Otherwise every episode is rejected for what it did not do."""
    llm = ScriptedLLM(['{"api": "venmo.create_transaction", "argument": "description",'
                       ' "predicate": "startswith", "value": "[personal]",'
                       ' "rule": "Start payment notes with the category tag."}'])
    u = TraceUpdater(llm)
    u.observe(_ep(), _fb())
    sms_only = EpisodeRecord(task_id="t", instruction="i", supervisor={},
                             api_calls=[ApiCall(app="phone", api="send_text_message",
                                                arguments={"message": "hi"})],
                             task_completed=True, phase="eval")
    assert u.check(sms_only) == []


def test_trace_rejects_an_unusable_check_instead_of_storing_it():
    """An unknown predicate must be counted as a parse failure, not silently stored as a rule
    that can never evaluate."""
    u = TraceUpdater(ScriptedLLM(['{"api": "phone.send_text_message", "argument": "message",'
                                  ' "predicate": "matches_vibe", "value": "x", "rule": "r"}']))
    u.observe(_ep(), _fb())
    assert u.state()["rules"] == 0 and u.state()["parse_failures"] == 1


def test_trace_skip_is_recorded_not_counted_as_failure():
    """The paper only compiles corrections concrete enough to test; a skip is a legitimate
    outcome and must be distinguishable from a malformed response."""
    u = TraceUpdater(ScriptedLLM(['{"skip": true}']))
    u.observe(_ep(), _fb())
    assert u.state()["skipped"] == 1 and u.state()["parse_failures"] == 0


# --- the multi-complaint contract, which the first live smoke run violated -------------------
# The instance tier emits ONE LINE PER VIOLATED RULE, so a single p13 episode carries up to five
# complaints and the writer LLM answers with an array. Taking only element [0] stored one
# precedent in five and TEPA finished a 3-episode run with 0 stored and 3 parse failures. These
# two tests pin the array contract so that regression cannot return silently.

def test_tepa_stores_every_precedent_in_a_multi_complaint_episode():
    llm = ScriptedLLM(['''[
      {"key": "payment_note.category", "content": "Notes must start with a category tag."},
      {"key": "payment_note.initials", "content": "Notes must end with initials."},
      {"key": "payment.visibility", "content": "Transfers must be private."},
      {"key": "sms.opening", "content": "Texts must open with a greeting."},
      {"key": "sms.ending", "content": "Texts must be signed with my first name."}
    ]'''])
    u = TepaUpdater(llm)
    u.observe(_ep(), _fb())
    st = u.state()
    assert st["active"] == 5, f"all five complaints should become precedents, got {st['active']}"
    assert st["keys"] == 5 and st["parse_failures"] == 0


def test_trace_compiles_every_check_in_a_multi_complaint_episode():
    llm = ScriptedLLM(['''[
      {"api": "venmo.create_transaction", "argument": "description",
       "predicate": "startswith", "value": "[personal]", "rule": "Tag payment notes."},
      {"api": "phone.send_text_message", "argument": "message",
       "predicate": "endswith", "value": "Lena", "rule": "Sign texts."}
    ]'''])
    u = TraceUpdater(llm)
    u.observe(_ep(), _fb())
    assert u.state()["rules"] == 2, "both compilable complaints should become checks"
    assert u.state()["parse_failures"] == 0


def test_trace_mixed_array_counts_skip_and_check_separately():
    """A skip alongside a real check must not be recorded as a parse failure."""
    llm = ScriptedLLM(['[{"skip": true}, {"api": "phone.send_text_message",'
                       ' "argument": "message", "predicate": "contains", "value": "hi",'
                       ' "rule": "Greet."}]'])
    u = TraceUpdater(llm)
    u.observe(_ep(), _fb())
    st = u.state()
    assert st["rules"] == 1 and st["skipped"] == 1 and st["parse_failures"] == 0


def test_trace_rejects_an_episode_quoted_value():
    """A check whose literal is copied from this episode's text must not be admitted.

    Seed 0 compiled `startswith 'ndations: Django Unchained, Pulp Fiction'` -- a mid-word fragment
    of one task's own message body. No other task can satisfy it, so it became a permanent gate
    that failed 28/28 and drove the arm's violation rate UP after n=12.
    """
    llm = ScriptedLLM(['{"api": "phone.send_text_message", "argument": "message",'
                       ' "predicate": "startswith",'
                       ' "value": "ndations: Django Unchained, Pulp Fiction",'
                       ' "rule": "Do not start SMS with movie recommendations."}'])
    u = TraceUpdater(llm)
    u.observe(_ep(), _fb())
    st = u.state()
    assert st["overlong_rejected"] == 1
    assert st["rules"] == 0, "an episode-quoted check must never become a compiled rule"


def test_trace_supports_negated_predicates():
    """Without a negation the writer inverts a prohibition into a positive requirement.

    Seed 0 compiled "Do not start SMS messages with 'lie, The Godfather...'" as `startswith` that
    same string -- a rule demanding the very text it was told to avoid.
    """
    llm = ScriptedLLM(['{"api": "phone.send_text_message", "argument": "message",'
                       ' "predicate": "not_startswith", "value": "Re:",'
                       ' "rule": "Do not start texts with Re:."}'])
    u = TraceUpdater(llm)
    u.observe(_ep(), _fb())
    assert u.state()["rules"] == 1

    bad = _ep_with_call("phone", "send_text_message", {"message": "Re: your order"})
    good = _ep_with_call("phone", "send_text_message", {"message": "Hi Sam, your order"})
    assert u.check(bad), "not_startswith must reject the forbidden prefix"
    assert not u.check(good), "not_startswith must pass text lacking the prefix"


def test_trace_treats_a_missing_argument_as_inapplicable():
    """A check naming an argument the call does not carry must not count as a violation.

    Seed 0 compiled `is_private equals true` while venmo's real argument is `private`. Treating the
    missing key as a failure made that check fail 65 of 68 times for a reason unrelated to the
    preference, which read as the arm regressing.
    """
    llm = ScriptedLLM(['{"api": "venmo.create_transaction", "argument": "is_private",'
                       ' "predicate": "equals", "value": "true",'
                       ' "rule": "Mark venmo transactions private."}'])
    u = TraceUpdater(llm)
    u.observe(_ep(), _fb())
    assert u.state()["rules"] == 1
    ep = _ep_with_call("venmo", "create_transaction", {"private": True, "description": "x"})
    assert not u.check(ep), "a rule naming an absent argument must be inapplicable, not failed"
    assert u.state()["arg_inapplicable"] == 1


def test_tepa_repromotion_keeps_one_active_per_key():
    """A re-promotion is a current-key write, so it must revoke the key's other active precedent.

    Seed 1 finished with active=6 over keys=5 because episode 17 wrote a new precedent under
    payment_note.ending and episode 18 re-promoted an older one under the same key, leaving both
    active. That breaks the invariant this class documents (the active set is bounded by the number
    of distinct keys) and quietly lets the injected memory grow.
    """
    llm = ScriptedLLM([
        '{"key": "payment_note.ending", "content": "notes must end with initials"}',
        '{"key": "payment_note.ending", "content": "notes must end with initials in brackets"}',
        '{"key": "payment_note.ending", "content": "notes must end with initials"}',
    ])
    u = TepaUpdater(llm)
    for _ in range(3):
        u.observe(_ep(), _fb())
    st = u.state()
    assert st["repromotions"] == 1, "the third write repeats the first, so it re-promotes"
    assert st["active"] == 1 == st["keys"], f"one active per key, got {st['active']}"
    assert "in brackets" not in u.render_memory(), "the superseded precedent must leave memory"
