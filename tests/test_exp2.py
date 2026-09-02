"""Unit tests for Exp-2 machinery: posterior replay + memory construction."""

from appworld_p.exp2 import Attribute, build_attributes, build_memory, replay_posterior
from appworld_p.rules import RULE_REGISTRY


def _attrs():
    return build_attributes(["sms_signoff", "venmo_private"], ["note_tags_required"])


def _event(accepted, verdicts, corrected=()):
    return {"accepted": accepted, "verdicts": verdicts, "corrected_rules": list(corrected),
            "task_id": "t"}


def test_build_attributes_requires_negation():
    attrs = _attrs()
    names = [a.rule_name for a in attrs]
    assert names == ["sms_signoff", "venmo_private", "note_tags_required"]
    assert attrs[0].is_persona and not attrs[2].is_persona


def test_corrective_replay_moves_posterior_up():
    attrs = _attrs()
    events = [_event(False, {"sms_signoff": False}, corrected=["sms_signoff"])] * 3
    post = replay_posterior(events, attrs, n=3, tier="corrective")
    sms = next(a for a in post if a.rule_name == "sms_signoff")
    assert sms.p_positive > 0.7
    # untouched attribute stays at prior
    other = next(a for a in post if a.rule_name == "note_tags_required")
    assert other.p_positive == 0.5


def test_default_tier_is_ambiguous_but_directional():
    attrs = _attrs()
    # rejection with two applicable rules, one satisfied one violated:
    events = [_event(False, {"sms_signoff": False, "venmo_private": True})] * 4
    post = replay_posterior(events, attrs, n=4, tier="default")
    sms = next(a for a in post if a.rule_name == "sms_signoff")
    venmo = next(a for a in post if a.rule_name == "venmo_private")
    assert sms.p_positive > 0.5        # violated + rejected -> wants it
    assert venmo.p_positive < 0.5      # satisfied + rejected -> maybe dislikes it (noise!)


def test_accept_gives_weak_consistent_evidence():
    attrs = _attrs()
    events = [_event(True, {"sms_signoff": True})] * 2
    post = replay_posterior(events, attrs, n=2, tier="corrective")
    sms = next(a for a in post if a.rule_name == "sms_signoff")
    assert 0.5 < sms.p_positive < 0.7  # weak positive


def test_replay_respects_n_prefix():
    attrs = _attrs()
    events = [_event(False, {"sms_signoff": False}, corrected=["sms_signoff"])] * 5
    early = replay_posterior(events, attrs, n=1, tier="corrective")
    late = replay_posterior(events, attrs, n=5, tier="corrective")
    get = lambda post: next(a for a in post if a.rule_name == "sms_signoff").p_positive
    assert get(late) > get(early) > 0.5


def test_memory_learned_uses_map_direction_and_L():
    attrs = _attrs()
    attrs[0].pos = 5           # sms_signoff strongly positive
    attrs[1].neg = 5           # venmo_private strongly NEGATIVE (wrong direction learned)
    memory = build_memory(attrs, L=2, variant="learned", seed=1)
    assert RULE_REGISTRY["sms_signoff"].oracle_text in memory
    assert RULE_REGISTRY["venmo_private"].negation_text in memory  # anti-assertion written!
    assert memory.count("- ") == 2
    memory1 = build_memory(attrs, L=1, variant="learned", seed=1)
    assert memory1.count("- ") == 1


def test_memory_oracle_variant_all_correct():
    attrs = _attrs()
    attrs[1].neg = 5  # even with wrong evidence, oracle writes the true positive text
    memory = build_memory(attrs, L=2, variant="oracle", seed=0)
    assert RULE_REGISTRY["venmo_private"].oracle_text in memory
    assert RULE_REGISTRY["venmo_private"].negation_text not in memory


def test_memory_padding_reaches_budget():
    attrs = _attrs()
    attrs[0].pos = 5
    short = build_memory(attrs, L=1, variant="learned", seed=0)
    padded = build_memory(attrs, L=1, variant="learned", seed=0, pad_to_chars=1500)
    assert len(padded) == 1500 and padded.startswith(short[:40])


def test_zero_evidence_direction_is_seeded_coin():
    attrs = _attrs()
    m1 = build_memory(attrs, L=3, variant="learned", seed=7)
    m2 = build_memory(attrs, L=3, variant="learned", seed=7)
    assert m1 == m2  # deterministic per seed


def test_memory_is_nested_in_L() -> None:
    """Raising L may only ADD assertions, never rewrite the ones already there.

    Tie-breaks used to be seeded with L, so an attribute with no evidence (posterior exactly
    0.5) got a fresh coin at every grid point and the same posterior produced different
    memories along the axis. The n=4 pilot curve -- 0.75, 0.625, 0.625, 0.125, 0.625, 0.25 --
    was those coins rather than thin evidence, which makes an L sweep uninterpretable: a rise
    could not be attributed to L at all.
    """
    from appworld_p.exp2 import build_attributes, build_memory

    attrs = build_attributes(["sms_signoff", "venmo_private"],
                             ["email_subject_short", "todoist_due_date"])
    for a in attrs:
        assert abs(a.p_positive - 0.5) < 1e-9, "precondition: no evidence, so every tie"

    prev: set[str] = set()
    for L in range(1, len(attrs) + 1):
        lines = {ln for ln in build_memory(attrs, L, seed=0).splitlines()
                 if ln.startswith("- ")}
        assert prev <= lines, f"L={L} dropped or rewrote {prev - lines}"
        assert len(lines) == L
        prev = lines


def test_tie_direction_is_stable_across_L() -> None:
    """A tied attribute keeps one direction for the whole sweep, so a reversal is a property
    of the evidence rather than of the grid point it was read at."""
    from appworld_p.exp2 import build_attributes, build_memory

    attrs = build_attributes(["sms_signoff"], ["email_subject_short", "todoist_due_date"])
    tied = attrs[0]
    seen = set()
    for L in range(1, len(attrs) + 1):
        mem = build_memory(attrs, L, seed=0)
        if tied.pos_text in mem:
            seen.add("pos")
        elif tied.neg_text in mem:
            seen.add("neg")
    assert len(seen) == 1, f"direction flipped across L: {seen}"
