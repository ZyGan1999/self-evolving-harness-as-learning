"""Family-B machinery: habit stream, external counter, usual_card checker."""

import pytest

from appworld_p.habit import (DEFAULT_PI_U, FAMILY_B_OPTIONS, HabitStream, pi_u_for)
from appworld_p.history import SessionHistory
from appworld_p.rules import RULE_REGISTRY
from conftest import call, make_episode

CARDS = {"11": "American Express", "12": "Wells Fargo", "13": "Chase"}


def stream(dominant="American Express", tier="binary", n=60, seed=0, **kw):
    h = HabitStream(pi_u=pi_u_for(dominant), tier=tier, seed=seed, **kw)
    h.advance(n)
    return h


# ------------------------------------------------------------------ habit stream

def test_pi_u_shapes():
    assert sum(DEFAULT_PI_U.values()) == pytest.approx(1.0)
    for dominant in FAMILY_B_OPTIONS:
        pi = pi_u_for(dominant)
        assert sum(pi.values()) == pytest.approx(1.0)
        assert max(pi, key=pi.get) == dominant
    with pytest.raises(ValueError):
        pi_u_for("Diners Club")


def test_target_is_pi_u_argmax_not_sample_dependent():
    """f*_u exists before the log does: the target must not move with n."""
    h = HabitStream(pi_u=pi_u_for("Wells Fargo"), seed=3)
    targets = []
    for n in (0, 5, 17, 60):
        h.advance(n)
        targets.append(h.target())
    assert targets == ["Wells Fargo"] * 4


def test_advance_is_monotone_and_deterministic():
    a, b = stream(n=30, seed=7), stream(n=30, seed=7)
    assert a.render_log() == b.render_log()
    a.advance(50)
    assert len(a.rounds) == 50
    assert a.render_log().startswith(b.render_log())   # prefix stability
    a.advance(10)                                     # never shrinks
    assert len(a.rounds) == 50


# ------------------------------------------------- information parity of the tiers

def test_tiers_differ_only_in_revelation():
    """Same underlying rounds; the binary tier must never name the wanted card."""
    corr, binr = stream(tier="corrective", n=40), stream(tier="binary", n=40)
    assert [(r.wanted, r.used) for r in corr.rounds] == [(r.wanted, r.used) for r in binr.rounds]
    assert "I wanted my" in corr.render_log()
    log = binr.render_log()
    assert "I wanted" not in log
    # a rejection line names the USED card and nothing else — no wanted-card identity
    for r, line in zip(binr.rounds, log.splitlines()):
        if not r.accepted:
            named = [o for o in FAMILY_B_OPTIONS if o in line]
            assert named == [r.used], f"{named} leaked in: {line}"
    assert len(log.splitlines()) == 40


def test_binary_log_is_faithful_to_rounds():
    h = stream(tier="binary", n=25)
    for r, line in zip(h.rounds, h.render_log().splitlines()):
        assert f"paid with {r.used} card" in line
        assert ("ok" in line) == r.accepted


def test_noise_interleaves_without_touching_habit_rounds():
    clean, noisy = stream(n=30, seed=5), stream(n=30, seed=5, noisy=True)
    assert len(noisy.rounds) == 30
    assert len(noisy.render_log().splitlines()) > len(clean.render_log().splitlines())
    assert noisy.wanted_counts() == clean.wanted_counts()   # same habit process


# ------------------------------------------------------------ external counter (control)

def test_counter_recovers_target_at_designed_n():
    """The control class must be reliable where the experiment reads it (n >= 40)."""
    for dominant in FAMILY_B_OPTIONS:
        hits = sum(stream(dominant=dominant, n=60, seed=100 + s).counter_argmax() == dominant
                   for s in range(20))
        assert hits >= 19, f"{dominant}: counter recovered {hits}/20 at n=60"


def test_counter_works_in_both_tiers():
    """Acceptance-rate credit assignment needs only used-card + accept/reject."""
    for seed in range(8):
        c = stream(tier="corrective", n=60, seed=seed).counter_argmax()
        b = stream(tier="binary", n=60, seed=seed).counter_argmax()
        assert c == b


def test_external_stats_names_one_card_and_no_ground_truth():
    h = stream(n=60)
    stats = h.external_stats()
    assert "accepted" in stats and h.counter_argmax() in stats
    assert "wanted" not in stats.lower()      # no leakage of per-round ground truth
    assert HabitStream(pi_u=pi_u_for("Chase")).external_stats() == ""   # empty at n=0


# ------------------------------------------------------------------ usual_card checker

def usual_card(calls, target="American Express", cards=CARDS):
    hist = SessionHistory(habit_target=target)
    ep = make_episode(calls, card_names=dict(cards))
    return RULE_REGISTRY["usual_card"]().check(ep, hist)


def pay(card_id=None, **kw):
    args = dict(receiver_email="sam@example.com", amount=20, **kw)
    if card_id is not None:
        args["payment_card_id"] = card_id
    return call("venmo", "create_transaction", url="/venmo/transactions", **args)


def test_usual_card_matches_habit():
    assert usual_card([pay(card_id=11)]).satisfied is True


def test_usual_card_wrong_bank_violates():
    res = usual_card([pay(card_id=12)])
    assert res.satisfied is False and "Wells Fargo" in res.detail


def test_usual_card_judges_first_attempt_only():
    """A card can fail for insufficient balance and force a fallback; intent is the
    first attempt, so a correct first card stays satisfied and a wrong one stays violated."""
    assert usual_card([pay(card_id=11), pay(card_id=12)]).satisfied is True
    assert usual_card([pay(card_id=12), pay(card_id=11)]).satisfied is False


def test_usual_card_balance_payment_violates():
    res = usual_card([pay()])
    assert res.satisfied is False and "balance" in res.detail


def test_usual_card_not_applicable_without_payment():
    res = usual_card([call("phone", "send_text_message", message="hi")])
    assert not res.applicable and res.satisfied is None


def test_usual_card_vacuous_before_habit_exists():
    res = usual_card([pay(card_id=12)], target=None)
    assert res.satisfied is True and "vacuous" in res.detail


def test_usual_card_hallucinated_id_is_a_violation():
    """Observed in the smoke run: the agent guesses payment_card_id=1 without ever
    listing the cards (real ids are 115-422). That is a real miss, not a lookup bug."""
    res = usual_card([pay(card_id=1)])
    assert res.satisfied is False and "unknown card id" in res.detail


def test_usual_card_unmeasurable_when_card_table_missing():
    """Our own lookup failing must not be recorded as a preference violation."""
    hist = SessionHistory(habit_target="American Express")
    ep = make_episode([pay(card_id=11)], card_names={})
    res = RULE_REGISTRY["usual_card"]().check(ep, hist)
    assert not res.applicable and res.satisfied is None
    # ... but a balance payment is still judged, table or no table
    ep2 = make_episode([pay()], card_names={})
    assert RULE_REGISTRY["usual_card"]().check(ep2, hist).satisfied is False


# ------------------------------------------------ family A hard rung: char checksum

def sms(message):
    return call("phone", "send_text_message", url="/phone/messages/text",
                phone_number="555-0102", message=message)


def test_sms_char_checksum_accepts_correct_count():
    body = "Please get on venmo."          # 20 chars -> 20 % 7 == 6
    res = RULE_REGISTRY["sms_char_checksum"]().check(
        make_episode([sms(f"{body} #6")]), SessionHistory())
    assert res.applicable and res.satisfied is True


def test_sms_char_checksum_rejects_word_count_answer():
    """The two rungs must disagree: 4 words -> #4, but 20 chars -> #6."""
    body = "Please get on venmo."
    ep = make_episode([sms(f"{body} #4")])
    assert RULE_REGISTRY["sms_char_checksum"]().check(ep, SessionHistory()).satisfied is False
    assert RULE_REGISTRY["sms_checksum"]().check(ep, SessionHistory()).satisfied is True


def test_sms_char_checksum_needs_the_tag():
    res = RULE_REGISTRY["sms_char_checksum"]().check(
        make_episode([sms("Please get on venmo.")]), SessionHistory())
    assert res.satisfied is False and "lacks checksum tag" in res.detail


def test_driver_scores_extra_rules_outside_the_persona():
    """calib_famA2 silently measured applicable=0/6 because sms_char_checksum was in the
    calibration list but not in p3_fc, so the driver never scored it."""
    from appworld_p.driver import SessionConfig, SessionDriver
    from appworld_p.updaters import NoOpUpdater
    from appworld_p.config import PERSONA_DIR
    cfg = SessionConfig(run_name="unit_extra_rules",
                        persona_path=str(PERSONA_DIR / "p3_fc.yaml"),
                        stream_task_ids=[], eval_task_ids=[], checkpoints=[0],
                        extra_rules=("sms_char_checksum",))
    names = [r.name for r in SessionDriver(cfg, NoOpUpdater()).persona.rules]
    assert "sms_char_checksum" in names
    assert names.count("sms_char_checksum") == 1      # idempotent, no duplicate scoring


def test_verifier_detail_appends_the_computed_value():
    """The reject-only gate exhausted 3 attempts on a 393-char message because every
    retry recomputed the same wrong count. verifier_detail hands the value back."""
    from appworld_p.config import PERSONA_DIR
    from appworld_p.driver import SessionConfig, SessionDriver
    from appworld_p.rules.base import RuleResult
    from appworld_p.updaters import NoOpUpdater

    viol = RuleResult("sms_char_checksum", applicable=True, satisfied=False,
                      detail="checksum wrong: got 5, expected 1 (393 chars)")

    def report(detail_on):
        cfg = SessionConfig(run_name="unit_vd", persona_path=str(PERSONA_DIR / "p4_q1.yaml"),
                            stream_task_ids=[], verifier_attempts=3,
                            verifier_detail=detail_on)
        return SessionDriver(cfg, NoOpUpdater())._linter_report([viol])

    plain, detailed = report(False), report(True)
    assert "REJECTED" in plain
    assert "expected 1" not in plain          # reject-only: the answer never leaks
    assert "expected 1" in detailed           # value-supplying: the answer is handed back
    assert "393 chars" in detailed


def test_every_declared_arm_dispatches():
    """oracle_verifier_value crashed 5h into seed 0 because `startswith("oracle")` caught
    it and int('_verifier_value') threw. Every arm the runner advertises must build."""
    import argparse
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    from run_exp1 import STATIC_ARMS, STREAM_ARMS, arm_config
    from appworld_p.config import PERSONA_DIR

    args = argparse.Namespace(n_train=4, n_eval=2, checkpoints=[0], seed=0,
                              feedback_tier="corrective", max_steps=30,
                              verifier_attempts=3, top_l=6, distractors=6,
                              agent="fc", llm="anthropic:m", persona="p4_q1")
    pool = {"eval_task_ids": ["t1", "t2"], "stream_task_ids": ["s1", "s2", "s3", "s4"]}
    for arm in sorted(STATIC_ARMS | STREAM_ARMS):
        cfg, _ = arm_config(arm, f"rn_{arm}", str(PERSONA_DIR / "p4_q1.yaml"), pool, args)
        assert cfg.memory_mode in {"none", "oracle", "updater"}, arm
        # a *_value arm must actually turn the flag on, or it silently duplicates its twin
        assert cfg.verifier_detail == arm.endswith("_value"), arm
        assert (cfg.verifier_attempts > 1) == ("verifier" in arm), arm


def test_n0_memory_does_not_already_contain_the_answer():
    """AssertionTopL sorts by posterior mean with a stable sort, so before any evidence the
    order is bank insertion order. The bank listed true rules first, so the learned arm's
    n=0 memory held both of them and the learning curve had nowhere to go."""
    from appworld_p.updaters import AssertionTopL, build_candidate_bank

    truth = ["private_note_format", "sms_char_checksum"]
    distractors = ["email_greeting_signoff", "email_subject_short", "todoist_due_date",
                   "todoist_title_capitalized", "file_kebab_case", "note_title_dated"]

    unshuffled = AssertionTopL(build_candidate_bank(truth, distractors), L=3)
    assert all(r in unshuffled.render_memory() for r in
               ("characters in the message", "K7 if the whole-dollar")), \
        "precondition: unshuffled bank leaks both true rules at n=0"

    # shuffled, no single seed may hand over the full answer set before training
    leaked = 0
    for seed in range(8):
        bank = build_candidate_bank(truth, distractors, shuffle_seed=seed)
        mem = AssertionTopL(bank, L=3).render_memory()
        if all(r in mem for r in ("characters in the message", "K7 if the whole-dollar")):
            leaked += 1
    assert leaked <= 2, f"{leaked}/8 seeds start with both true rules already in memory"
