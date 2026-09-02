"""Exp 2 (Q2) online pool collection: the learner INDUCES its own memory lines.

Differs from run_exp2_collect.py in what memory is made of. There, memory lines were
oracle_text / negation_text -- hand-written, calibrated to 1.00 compliance -- so L > d had
to be filled with distractor rules. Here every line is written by the learner from an
instance-tier complaint, so all L lines describe the same d preferences and no distractor
bank is involved: L > d is redundancy about the persona, not contamination from outside it.

Memory stays empty during collection (fixed data-collection policy), so the pool is a
function of the stream only. The L-sweep then selects from the finished pool offline.

Usage:
  python scripts/run_exp2_pool.py --llm anthropic:claude-haiku-4-5-20251001 \
      --persona p6_q2 --n-train 32 [--seed 1] [--tag _v1]
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from appworld_p.config import OUTPUTS_DIR, PERSONA_DIR, TASK_POOL_PATH  # noqa: E402
from appworld_p.driver import SessionConfig, SessionDriver  # noqa: E402
from appworld_p.llm import build_llm  # noqa: E402
from appworld_p.persona import Persona  # noqa: E402
from appworld_p.streams import build_stream  # noqa: E402
from appworld_p.summarize import AssertionCollector  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm", required=True)
    parser.add_argument("--persona", default="p6_q2")
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--agent", default="fc", choices=["fc", "react"])
    parser.add_argument("--n-train", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tag", default="")
    # The learner that writes memory. Same model as the agent by default: a stronger
    # summariser would be a different harness, not a different L.
    parser.add_argument("--learner-llm", default="")
    args = parser.parse_args()

    pool = json.loads(TASK_POOL_PATH.read_text())["pools"][args.persona]
    persona = Persona.load(PERSONA_DIR / f"{args.persona}.yaml")
    model_slug = args.llm.split(":")[-1].replace(".", "_")
    run_name = f"exp2pool{args.tag}_{model_slug}_{args.persona}_s{args.seed}"

    collector = AssertionCollector(build_llm(args.learner_llm or args.llm))
    config = SessionConfig(
        run_name=run_name,
        persona_path=str(PERSONA_DIR / f"{args.persona}.yaml"),
        stream_task_ids=build_stream(pool, args.n_train, args.seed),
        eval_task_ids=[], checkpoints=[],
        agent=args.agent, llm=args.llm, seed=args.seed,
        max_steps=args.max_steps,
        feedback_tier="instance",     # complaint about THIS episode, no general rule given
        memory_mode="updater",        # AssertionCollector renders "" -> fixed policy
        notes="exp2 online pool collection (learner-induced assertions, instance feedback)",
    )
    SessionDriver(config, collector).run()

    out = OUTPUTS_DIR / f"exp2pool{args.tag}_{model_slug}_{args.persona}_s{args.seed}.json"
    out.write_text(json.dumps({
        "llm": args.llm, "learner_llm": args.learner_llm or args.llm,
        "persona": args.persona, "seed": args.seed, "n_train": args.n_train,
        "persona_rules": [r.name for r in persona.rules],
        "assertions": [a.__dict__ for a in collector.assertions],
        "events": collector.events,
    }, ensure_ascii=False, indent=1))
    print(f"Wrote {out} ({len(collector.assertions)} assertions "
          f"over {collector.state()['episodes']} episodes)")


if __name__ == "__main__":
    main()
