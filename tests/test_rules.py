"""Positive/negative cases for every checker (R003 quality gate)."""

import tempfile
from pathlib import Path

import pytest

from appworld_p.history import SessionHistory
from appworld_p.rules import RULE_REGISTRY, build_rules
from conftest import call, make_episode


def check(rule_name: str, calls, history=None, **ep_kwargs):
    rule = RULE_REGISTRY[rule_name]()
    ep = make_episode(calls, **ep_kwargs)
    return rule.check(ep, history or SessionHistory())


# ---------------------------------------------------------------- pool A

GOOD_EMAIL = dict(subject="Lunch plan", body="Hi Sam,\nSee you at noon.\nBest,\nLena")


def test_email_greeting_signoff():
    ok = check("email_greeting_signoff", [call("gmail", "send_email", **GOOD_EMAIL)])
    assert ok.applicable and ok.satisfied
    bad = check("email_greeting_signoff",
                [call("gmail", "send_email", subject="x", body="See you at noon.")])
    assert bad.satisfied is False
    na = check("email_greeting_signoff", [call("venmo", "create_transaction", amount=5)])
    assert not na.applicable and na.satisfied is None


def test_email_subject_short():
    ok = check("email_subject_short", [call("gmail", "send_email", **GOOD_EMAIL)])
    assert ok.satisfied
    bad = check("email_subject_short", [call(
        "gmail", "send_email", subject="a very long subject line that keeps on going forever",
        body="x")])
    assert bad.satisfied is False


def test_todoist_due_date():
    ok = check("todoist_due_date",
               [call("todoist", "create_task", title="Buy milk", due_date="2026-07-08")])
    assert ok.satisfied
    bad = check("todoist_due_date", [call("todoist", "create_task", title="Buy milk")])
    assert bad.satisfied is False


def test_todoist_title_capitalized():
    ok = check("todoist_title_capitalized",
               [call("todoist", "create_task", title="Buy milk")])
    assert ok.satisfied
    bad = check("todoist_title_capitalized",
                [call("todoist", "create_task", title="buy milk")])
    assert bad.satisfied is False


def test_file_kebab_case():
    ok = check("file_kebab_case",
               [call("file_system", "create_file", name="trip-notes-2026.txt")])
    assert ok.satisfied
    bad = check("file_kebab_case",
                [call("file_system", "create_file", name="Trip Notes.txt")])
    assert bad.satisfied is False


def test_note_title_dated():
    ok = check("note_title_dated",
               [call("simple_note", "create_note", title="2026-07-07: groceries", content="x")])
    assert ok.satisfied
    bad = check("note_title_dated",
                [call("simple_note", "create_note", title="groceries", content="x")])
    assert bad.satisfied is False


def test_payment_has_note():
    ok = check("payment_has_note",
               [call("venmo", "create_transaction", url="/venmo/transactions",
                     amount=12.5, description="dinner")])
    assert ok.satisfied
    bad = check("payment_has_note",
                [call("venmo", "create_transaction", url="/venmo/transactions", amount=12.5)])
    assert bad.satisfied is False


def test_playlist_private():
    ok = check("playlist_private",
               [call("spotify", "create_playlist", url="/spotify/playlists",
                     title="workout", is_public=False)])
    assert ok.satisfied
    default_private = check("playlist_private",
                            [call("spotify", "create_playlist", url="/spotify/playlists",
                                  title="workout")])
    assert default_private.satisfied
    bad = check("playlist_private",
                [call("spotify", "create_playlist", url="/spotify/playlists",
                      title="workout", is_public=True)])
    assert bad.satisfied is False


def test_playlist_over_like():
    ok = check("playlist_over_like",
               [call("spotify", "add_song_to_playlist", url="/spotify/playlists/1/songs",
                     song_id=9)])
    assert ok.applicable and ok.satisfied
    bad = check("playlist_over_like",
                [call("spotify", "like_song", url="/spotify/songs/9/like", song_id=9)])
    assert bad.satisfied is False


# ---------------------------------------------------------------- pool B

def _venmo(amount, description):
    return call("venmo", "create_transaction", url="/venmo/transactions",
                amount=amount, description=description)


def test_private_note_format():
    # whole-DOLLAR parity (cents parity is degenerate on integer-dollar tasks)
    ok = check("private_note_format", [_venmo(12.50, "[K7|2026-07-07|dinner split]")])
    assert ok.satisfied, ok.detail  # 12 dollars, even -> K7
    wrong_code = check("private_note_format", [_venmo(91, "[K7|2026-07-07|dinner split]")])
    assert wrong_code.satisfied is False  # 91 odd -> Q3 expected
    ok_odd = check("private_note_format", [_venmo(91, "[Q3|2026-07-07|dinner split]")])
    assert ok_odd.satisfied, ok_odd.detail
    plain = check("private_note_format", [_venmo(10, "dinner")])
    assert plain.satisfied is False
    long_summary = check("private_note_format",
                         [_venmo(10, "[K7|2026-07-07|one two three four five six seven]")])
    assert long_summary.satisfied is False


def test_recent_mode_card():
    history = SessionHistory(payment_card_choices=["c1", "c2", "c1", "c1", "c3"])
    pay = call("venmo", "create_transaction", url="/venmo/transactions",
               amount=5, payment_card_id="c1")
    ok = RULE_REGISTRY["recent_mode_card"]().check(make_episode([pay]), history)
    assert ok.satisfied, ok.detail
    wrong = call("venmo", "create_transaction", url="/venmo/transactions",
                 amount=5, payment_card_id="c2")
    bad = RULE_REGISTRY["recent_mode_card"]().check(make_episode([wrong]), history)
    assert bad.satisfied is False
    # vacuous with no history
    fresh = RULE_REGISTRY["recent_mode_card"]().check(make_episode([pay]), SessionHistory())
    assert fresh.satisfied
    # not applicable when payment made without a card
    nocard = RULE_REGISTRY["recent_mode_card"]().check(make_episode([_venmo(5, "x")]), history)
    assert not nocard.applicable


def test_txn_running_count():
    history = SessionHistory(payment_count=16)
    ok = RULE_REGISTRY["txn_running_count"]().check(
        make_episode([_venmo(5, "coffee txn #17")]), history)
    assert ok.satisfied, ok.detail
    bad = RULE_REGISTRY["txn_running_count"]().check(
        make_episode([_venmo(5, "coffee txn #9")]), history)
    assert bad.satisfied is False
    missing = RULE_REGISTRY["txn_running_count"]().check(
        make_episode([_venmo(5, "coffee")]), history)
    assert missing.satisfied is False


def test_checksum_subject():
    body = "one two three four five six seven eight nine"  # 9 words -> 9 % 7 = 2
    ok = check("checksum_subject",
               [call("gmail", "send_email", subject="Plan #2", body=body)])
    assert ok.satisfied, ok.detail
    bad = check("checksum_subject",
                [call("gmail", "send_email", subject="Plan #3", body=body)])
    assert bad.satisfied is False
    untagged = check("checksum_subject",
                     [call("gmail", "send_email", subject="Plan", body=body)])
    assert untagged.satisfied is False


def test_note_exact_words():
    words_25 = " ".join(f"w{i}" for i in range(25))
    ok = check("note_exact_words",
               [call("simple_note", "create_note", title="t", content=words_25)])
    assert ok.satisfied, ok.detail
    bad = check("note_exact_words",
                [call("simple_note", "create_note", title="t", content="too short")])
    assert bad.satisfied is False


def test_lipogram_email_close():
    ok = check("lipogram_email_close",
               [call("gmail", "send_email", subject="s",
                     body="Hello Sam,\nDetails attached.\nAll good, saying so long now.")])
    assert ok.satisfied, ok.detail
    bad = check("lipogram_email_close",
                [call("gmail", "send_email", subject="s", body="Hi,\nSee you later.")])
    assert bad.satisfied is False


def test_sms_signoff():
    ok = check("sms_signoff",
               [call("phone", "send_text_message", url="/phone/messages/text/555",
                     phone_number="555", message="Running late, be there soon. - Lena")])
    assert ok.satisfied, ok.detail
    bad = check("sms_signoff",
                [call("phone", "send_text_message", url="/phone/messages/text/555",
                      phone_number="555", message="Running late.")])
    assert bad.satisfied is False


def test_venmo_private():
    ok = check("venmo_private",
               [call("venmo", "create_transaction", url="/venmo/transactions",
                     amount=5, description="x", private=True)])
    assert ok.satisfied
    bad = check("venmo_private",
                [call("venmo", "create_transaction", url="/venmo/transactions",
                      amount=5, description="x")])
    assert bad.satisfied is False


def test_note_tags_required():
    ok = check("note_tags_required",
               [call("simple_note", "create_note", title="t", content="x", tags=["home"])])
    assert ok.satisfied
    bad = check("note_tags_required",
                [call("simple_note", "create_note", title="t", content="x")])
    assert bad.satisfied is False


def test_sms_checksum():
    msg = "one two three four five six seven eight nine"  # 9 words -> 2
    ok = check("sms_checksum",
               [call("phone", "send_text_message", url="/phone/messages/text/555",
                     phone_number="555", message=f"{msg} #2")])
    assert ok.satisfied, ok.detail
    bad = check("sms_checksum",
                [call("phone", "send_text_message", url="/phone/messages/text/555",
                      phone_number="555", message=f"{msg} #5")])
    assert bad.satisfied is False


# ---------------------------------------------------------------- framework

def test_field_claim_conflict_detected():
    with pytest.raises(ValueError, match="field claim conflict"):
        build_rules(["email_greeting_signoff", "lipogram_email_close"])


def test_registry_complete():
    pools = {RULE_REGISTRY[n].pool for n in RULE_REGISTRY}
    assert pools == {"A", "B"}
    # private_note_format moved B -> A: calibration measured oracle 1.00 / baseline 0.00.
    # +4 for the Q2 attribute slots (file_overwrite_always, sms_greeting,
    # payment_note_initials, file_content_header), which exist because the pool-A rules
    # already present are mostly base-saturated -- their empty-memory compliance leaves no
    # room for a correct assertion to help, so they cannot carry a descending branch.
    # +2 for the Q3 implicit-threshold rules (payment_note_brief, sms_not_terse), which state a
    # numeric bound in oracle_text and withhold it from the complaint. They are registered but
    # unused: measurement showed payment_note_brief has no headroom (the agent already writes
    # short notes, 0/8 on the none arm) and sms_not_terse's bound fights the task-dictated body,
    # so p13 carries the withheld-literal rules instead (docs/Q3.md 3.6).
    assert sum(1 for n in RULE_REGISTRY if RULE_REGISTRY[n].pool == "A") == 27
    # +2: the spend_total_* rungs, exact aggregation over the real transaction history
    assert sum(1 for n in RULE_REGISTRY if RULE_REGISTRY[n].pool == "B") == 10
    for name, cls in RULE_REGISTRY.items():
        assert cls.oracle_text and cls.correction_template and cls.apps, name
        assert cls.trigger_actions, f"{name} lacks trigger_actions"
        assert cls().trigger_apis(), f"{name} trigger_apis resolves to nothing"


def test_api_rejected_calls_are_not_scored():
    """calib_famA3: the FC agent guessed message_body=, the API answered 'Validation
    error: message: field required', then it retried correctly. Scoring the rejected
    attempt invented a violation for a message that WAS sent with a tag."""
    import json
    from appworld_p.apicalls import load_api_calls
    tok = "t0k"
    rejected = {"method": "post", "url": "/phone/messages/text/555",
                "data": {"access_token": tok, "message_body": "Please get on venmo. #1"}}
    accepted = {"method": "post", "url": "/phone/messages/text/555",
                "data": {"access_token": tok, "message": "Please get on venmo. #1"}}
    p = Path(tempfile.mkdtemp()) / "api_calls.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in (rejected, accepted)))
    kept = load_api_calls(p)
    assert len(kept) == 1, [c.arguments for c in kept]
    assert kept[0].arg("message") == "Please get on venmo. #1"
    assert len(load_api_calls(p, keep_rejected=True)) == 2


# ------------------------------------------------- exact aggregation (spend totals)

SPEND = {"total": 5205.0, "count": 95,
         "per_recipient": {"connorbrow@gmail.com": 1153.0, "saritt@gmail.com": 1035.0}}


def pay(amount, note, to="connorbrow@gmail.com"):
    return call("venmo", "create_transaction", amount=amount,
                receiver_email=to, description=note)


def test_spend_total_per_recipient():
    ok = check("spend_total_per_recipient", [pay(25, "Dinner total: $1178")],
               spend_totals=SPEND)
    assert ok.applicable and ok.satisfied, ok.detail
    off_by_one = check("spend_total_per_recipient", [pay(25, "Dinner total: $1177")],
                       spend_totals=SPEND)
    assert off_by_one.satisfied is False
    # a first-time recipient starts from zero rather than from the all-time total
    fresh = check("spend_total_per_recipient",
                  [pay(40, "Cab total: $40", to="newperson@gmail.com")], spend_totals=SPEND)
    assert fresh.satisfied, fresh.detail
    missing = check("spend_total_per_recipient", [pay(25, "Dinner")], spend_totals=SPEND)
    assert missing.satisfied is False
    # no ground truth read -> instrumentation gap, judged neither way
    na = check("spend_total_per_recipient", [pay(25, "Dinner total: $1178")])
    assert not na.applicable and na.satisfied is None


def test_spend_total_all_time():
    ok = check("spend_total_all_time", [pay(25, "Dinner total: $5230")], spend_totals=SPEND)
    assert ok.applicable and ok.satisfied, ok.detail
    wrong = check("spend_total_all_time", [pay(25, "Dinner total: $5205")], spend_totals=SPEND)
    assert wrong.satisfied is False


def test_spend_totals_accumulate_within_one_episode():
    """Two payments in one episode: the second note must include the first."""
    two = check("spend_total_all_time",
                [pay(25, "One total: $5230"), pay(10, "Two total: $5240")],
                spend_totals=SPEND)
    assert two.satisfied, two.detail
    stale = check("spend_total_all_time",
                  [pay(25, "One total: $5230"), pay(10, "Two total: $5215")],
                  spend_totals=SPEND)
    assert stale.satisfied is False
    # per-recipient keeps separate tallies, so a second payee does not inherit the first
    split = check("spend_total_per_recipient",
                  [pay(25, "A total: $1178"), pay(15, "B total: $1050", to="saritt@gmail.com")],
                  spend_totals=SPEND)
    assert split.satisfied, split.detail


def test_spend_total_cents_tolerated():
    ok = check("spend_total_all_time", [pay(25.5, "x total: $5230.50")], spend_totals=SPEND)
    assert ok.satisfied, ok.detail


def test_spend_total_detail_names_every_payment():
    """The verifier_value gate hands this detail back, so one expectation is not enough.

    Two payments to the same person need two different running totals. Reporting only the
    first left the agent rewriting one note and failing on the next attempt -- p5_q1b s0
    burned all three gate attempts on 37a8675_1 that way.
    """
    both_wrong = check("spend_total_per_recipient",
                       [pay(25, "One total: $1"), pay(10, "Two total: $2")],
                       spend_totals=SPEND)
    assert both_wrong.satisfied is False
    d = both_wrong.detail
    assert "payment 1" in d and "payment 2" in d, d
    # the correct figures for BOTH payments must be present: 1153+25, then +10
    assert "1178" in d and "1188" in d, d
    # a missing tag still gets told what to write, not just that something is absent
    missing = check("spend_total_per_recipient", [pay(25, "Dinner")], spend_totals=SPEND)
    assert missing.satisfied is False
    assert "1178" in missing.detail, missing.detail


# ----------------------------------------------------- Q2 attribute slots (new pool A rules)

def test_file_overwrite_always():
    ok = check("file_overwrite_always",
               [call("file_system", "create_file", file_path="/a.csv", overwrite=True)])
    assert ok.applicable and ok.satisfied
    for bad_flag in (False, None, "no"):
        bad = check("file_overwrite_always",
                    [call("file_system", "create_file", file_path="/a.csv",
                          **({"overwrite": bad_flag} if bad_flag is not None else {}))])
        assert bad.satisfied is False, bad_flag
    # the API reports booleans as strings in some traces, so accept the string form
    assert check("file_overwrite_always",
                 [call("file_system", "create_file", file_path="/a.csv",
                       overwrite="True")]).satisfied
    na = check("file_overwrite_always", [call("phone", "send_text_message", message="hi")])
    assert not na.applicable and na.satisfied is None


def test_sms_greeting():
    ok = check("sms_greeting", [call("phone", "send_text_message",
                                     message="Hi Sam, the report is ready. Lena")])
    assert ok.applicable and ok.satisfied
    bad = check("sms_greeting", [call("phone", "send_text_message",
                                      message="The report is ready.")])
    assert bad.satisfied is False
    # greeting and sign-off occupy different ends of the same string and must not collide
    both_ok = check("sms_signoff", [call("phone", "send_text_message",
                                         message="Hi Sam, the report is ready. Lena")],
                    supervisor={"first_name": "Lena"})
    assert both_ok.satisfied


def test_payment_note_initials():
    ok = check("payment_note_initials",
               [call("venmo", "create_transaction", amount=5, description="Dinner (J.D.)")])
    assert ok.applicable and ok.satisfied
    for note in ("Dinner", "Dinner (JD)", "(J.D.) Dinner"):
        bad = check("payment_note_initials",
                    [call("venmo", "create_transaction", amount=5, description=note)])
        assert bad.satisfied is False, note


def test_file_content_header():
    ok = check("file_content_header",
               [call("file_system", "create_file", file_path="/a.csv",
                     content="# generated by assistant\nsong,album")])
    assert ok.applicable and ok.satisfied
    bad = check("file_content_header",
                [call("file_system", "create_file", file_path="/a.csv",
                      content="song,album")])
    assert bad.satisfied is False
    # no content written -> nothing to constrain, so not applicable rather than a violation
    na = check("file_content_header",
               [call("file_system", "create_file", file_path="/a.csv")])
    assert not na.applicable and na.satisfied is None


def test_payment_note_category():
    ok = check("payment_note_category",
               [call("venmo", "create_transaction", amount=5,
                     description="[personal] Dinner")])
    assert ok.applicable and ok.satisfied
    for note in ("Dinner", "Dinner [personal]", "(personal) Dinner"):
        bad = check("payment_note_category",
                    [call("venmo", "create_transaction", amount=5, description=note)])
        assert bad.satisfied is False, note


def test_note_prefix_and_suffix_coexist():
    """The prefix and suffix rules share venmo's description but claim opposite ends, which is
    what lets Q2 get two attributes out of one field."""
    both = "[personal] Dinner (J.D.)"
    args = [call("venmo", "create_transaction", amount=5, description=both)]
    assert check("payment_note_category", args).satisfied
    assert check("payment_note_initials", args).satisfied
    build_rules(["payment_note_category", "payment_note_initials"])   # no claim conflict


# ------------------------------------- interfering distractors (Q2 rising-branch mechanism)

def test_interfering_distractors_conflict_with_the_scored_rules():
    """These exist to COST something when written, so each must be violated by exactly the text
    the persona's own rules require. A distractor that the persona's output satisfies anyway
    could never produce a rising branch -- that was the flaw in the first Q2 bank, where every
    distractor sat on an app no scored rule touched.
    """
    compliant_note = "[personal] Dinner (J.D.)"          # satisfies both persona note rules
    note_args = [call("venmo", "create_transaction", amount=5, description=compliant_note)]
    for rule in ("payment_note_single_word", "payment_note_no_brackets",
                 "payment_note_lowercase"):
        assert check(rule, note_args).satisfied is False, rule
    private = [call("venmo", "create_transaction", amount=5, description=compliant_note,
                    private=True)]
    assert check("venmo_public_feed", private).satisfied is False

    compliant_sms = "Hi Sam, the quarterly report is attached for you. Lena"
    sms_args = [call("phone", "send_text_message", message=compliant_sms)]
    for rule in ("sms_terse", "sms_no_names", "sms_no_punctuation"):
        assert check(rule, sms_args).satisfied is False, rule
    # and the persona's own rules ARE satisfied by that same text, so the conflict is real
    assert check("sms_greeting", sms_args).satisfied
    assert check("sms_signoff", sms_args, supervisor={"first_name": "Lena"}).satisfied


def test_interfering_distractors_have_disjoint_claims():
    """They must be composable with the persona for the bank to hold them all at once."""
    build_rules(["sms_signoff", "sms_greeting", "venmo_private", "payment_note_initials",
                 "payment_note_category", "payment_note_single_word",
                 "payment_note_no_brackets", "sms_terse", "payment_has_note",
                 "venmo_public_feed", "sms_no_names", "payment_note_lowercase",
                 "sms_no_punctuation"])


def test_every_scored_rule_has_an_interfering_counterpart():
    """A scored rule with no counterpart cannot be harmed however large L gets, so it only ever
    contributes to the descending branch. venmo_private was in that position for the first run.
    """
    scored_to_distractor = {
        "sms_greeting": ("sms_no_names", "sms_no_punctuation", "sms_terse"),
        "sms_signoff": ("sms_no_names", "sms_no_punctuation", "sms_terse"),
        "venmo_private": ("venmo_public_feed",),
        "payment_note_initials": ("payment_note_lowercase", "payment_note_single_word",
                                  "payment_has_note"),
        "payment_note_category": ("payment_note_no_brackets", "payment_note_single_word",
                                  "payment_has_note"),
    }
    for scored, others in scored_to_distractor.items():
        assert scored in RULE_REGISTRY
        for other in others:
            assert other in RULE_REGISTRY, other
            assert RULE_REGISTRY[other].field_claims != RULE_REGISTRY[scored].field_claims
