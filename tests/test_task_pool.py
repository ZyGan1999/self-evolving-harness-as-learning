"""Eval-set construction properties."""
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from build_task_pool import build_pool_for_persona  # noqa: E402

from appworld_p.persona import Persona  # noqa: E402
from appworld_p.rules import RULE_REGISTRY  # noqa: E402


def _tasks(spec):
    """spec: [(task_id, [apis])] -> task metadata dicts."""
    return [{"task_id": tid, "difficulty": 1, "required_apis": apis} for tid, apis in spec]


def test_eval_set_spreads_across_task_templates():
    """'530b157_1/_2/_3' are variants of ONE template. A template that triggers both
    rules used to win every greedy pick and take 3 of 4 eval slots for the sms rule,
    which measures that template rather than the rule."""
    persona = Persona(name="t", rules=[RULE_REGISTRY["private_note_format"](),
                                       RULE_REGISTRY["sms_char_checksum"]()])
    sms, venmo = "phone.send_text_message", "venmo.create_transaction"
    tasks = _tasks(
        [(f"both{i}_{v}", [sms, venmo]) for i in range(2) for v in (1, 2, 3)]
        + [(f"sms{i}_{v}", [sms]) for i in range(2) for v in (1, 2, 3)]
        + [(f"ven{i}_{v}", [venmo]) for i in range(2) for v in (1, 2, 3)]
    )
    pool = build_pool_for_persona(persona, tasks, max_difficulty=3, eval_size=8,
                                  min_cover=4, rng=random.Random(0))
    fams = {t.split("_")[0] for t in pool["eval_task_ids"]}
    assert len(fams) >= 4, pool["eval_task_ids"]
    # and no single template may supply more than half the eval set
    counts = {f: sum(1 for t in pool["eval_task_ids"] if t.startswith(f)) for f in fams}
    assert max(counts.values()) <= len(pool["eval_task_ids"]) / 2, counts


def test_eval_and_stream_never_overlap():
    """A stream task seen during learning must not reappear at eval."""
    persona = Persona(name="t", rules=[RULE_REGISTRY["sms_char_checksum"]()])
    tasks = _tasks([(f"f{i}_{v}", ["phone.send_text_message"])
                    for i in range(4) for v in (1, 2, 3)])
    pool = build_pool_for_persona(persona, tasks, max_difficulty=3, eval_size=6,
                                  min_cover=6, rng=random.Random(1))
    assert not (set(pool["eval_task_ids"]) & set(pool["stream_task_ids"]))
