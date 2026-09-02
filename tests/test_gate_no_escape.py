"""A retry gate must not be satisfiable by declining to act.

Observed in the habitual-card gate arm: rejected on attempt 1, the agent burned attempt 2's step
budget re-reading the card list and never sent the payment. No rule was applicable, so no rule
was violated, and the loop banked that attempt as a pass -- the episode then left the denominator
as "inapplicable" rather than counting as a miss, which makes the arm look better the more often
the agent gives up. These tests pin the two behaviours that fix it: a dodge is retried rather
than accepted, and if the retries never comply, the attempt that acted and failed is the one
recorded.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from appworld_p.driver import SessionConfig, SessionDriver  # noqa: E402
from appworld_p.episode import EpisodeRecord  # noqa: E402
from appworld_p.rules.base import RuleResult  # noqa: E402


class _Persona:
    """Scripts one verdict per attempt: 'miss' acted and failed, 'dodge' never acted."""

    rules: list = []

    def __init__(self, script: list[str]) -> None:
        self.script = script

    def check(self, ep: EpisodeRecord, _history) -> list[RuleResult]:
        kind = self.script[ep.meta["scripted_index"]]
        if kind == "dodge":
            return [RuleResult(rule="r", applicable=False, satisfied=None, detail="")]
        return [RuleResult(rule="r", applicable=True, satisfied=kind == "pass", detail="d")]

    def oracle_context(self, _v: int) -> str:
        return ""


def _driver(script: list[str], attempts: int = 3) -> tuple[SessionDriver, list[str]]:
    """Driver whose episode runner is stubbed, so only the retry-selection logic is exercised."""
    cfg = SessionConfig(run_name="t", persona_path="t", stream_task_ids=[],
                        verifier_attempts=attempts, verifier_feedback=True)
    drv = SessionDriver.__new__(SessionDriver)
    drv.config = cfg
    drv.persona = _Persona(script)
    drv.history = None
    drv._correction_templates = {"r": "fix it"}
    order: list[str] = []

    def fake_once(task_id, phase, session_index, experiment_name, extra_memory=""):
        idx = len(order)
        order.append(script[idx])
        ep = EpisodeRecord(task_id=task_id, instruction="", supervisor={}, api_calls=[],
                           task_completed=True, phase=phase, session_index=session_index)
        ep.meta["scripted_index"] = idx
        return ep

    drv._run_episode_once = fake_once
    return drv, order


def test_dodge_is_retried_not_banked() -> None:
    # attempt 1 acts and misses, attempt 2 dodges, attempt 3 complies -> must reach attempt 3
    drv, order = _driver(["miss", "dodge", "pass"])
    ep = drv._run_episode("task", "eval", 0, "exp")
    assert order == ["miss", "dodge", "pass"], f"stopped early at the dodge: {order}"
    assert ep.meta["verifier_attempt"] == 3
    assert ep.meta["gate_escapes"] == 1


def test_acting_miss_outranks_a_dodge() -> None:
    # nothing ever complies: the recorded episode must be the one that acted, so the miss is
    # counted instead of vanishing from the denominator
    drv, _ = _driver(["miss", "dodge", "dodge"])
    ep = drv._run_episode("task", "eval", 0, "exp")
    verdicts = drv.persona.check(ep, None)
    assert verdicts[0].applicable is True, "a dodge was recorded over an attempt that acted"
    assert verdicts[0].satisfied is False
    assert ep.meta["gate_escapes"] == 2


def test_single_attempt_is_untouched() -> None:
    # attempts=1 is the no-gate configuration every non-gate arm runs under; it must keep
    # returning the first episode with no gate bookkeeping at all
    drv, order = _driver(["dodge"], attempts=1)
    ep = drv._run_episode("task", "eval", 0, "exp")
    assert order == ["dodge"]
    assert "gate_escapes" not in ep.meta and "verifier_attempt" not in ep.meta


def test_accumulated_report_keeps_one_header() -> None:
    """Exclusions accumulate under a single header, oldest first.

    Concatenating whole reports instead repeated "checker REJECTED the previous attempt" once per
    attempt, so a third try read as three separate verdicts on the same attempt rather than one
    verdict listing three excluded options.
    """
    from appworld_p.driver import REJECT_HEADER

    seen_prompts: list[str] = []
    drv, _ = _driver(["miss", "miss", "miss"])
    drv.config.verifier_accumulate = True
    drv._correction_templates = {"r": "fix it"}
    inner = drv._run_episode_once

    def spy(task_id, phase, session_index, experiment_name, extra_memory=""):
        seen_prompts.append(extra_memory)
        return inner(task_id, phase, session_index, experiment_name, extra_memory)

    drv._run_episode_once = spy
    drv._run_episode("task", "eval", 0, "exp")

    assert seen_prompts[0] == ""                       # first attempt is unprompted
    assert seen_prompts[2].count(REJECT_HEADER.strip().splitlines()[0]) == 1
    # the identical violation repeats, so dedupe should leave exactly one bullet
    assert seen_prompts[2].count("- fix it") == 1
