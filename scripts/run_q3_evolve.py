"""Run Q3 with ACE, Reflexion, TEPA, TRACE, and the reported reference and diagnostic arms."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from appworld_p.ace_updater import ACEStyleUpdater  # noqa: E402
from appworld_p.baseline_updaters import (ReflexionUpdater, TepaUpdater, TraceUpdater)  # noqa: E402
from appworld_p.config import OUTPUTS_DIR, PERSONA_DIR, TASK_POOL_PATH  # noqa: E402
from appworld_p.driver import SessionConfig, SessionDriver  # noqa: E402
from appworld_p.llm import build_llm  # noqa: E402
from appworld_p.streams import build_stream  # noqa: E402
from appworld_p.updaters import NoOpUpdater, SelfEvolveUpdater  # noqa: E402


EVOLVING_ARMS = {"selfevolve_rewrite", "selfevolve_ace", "external_corrective",
                 "reflexion", "tepa", "trace"}


def arm_config(arm, run_name, persona_path, pool, args):
    """(SessionConfig, updater) for one arm."""
    eval_ids = pool["eval_task_ids"][: args.n_eval]


    eval_ids = [t for t in eval_ids for _ in range(args.rollouts)]
    base = dict(run_name=run_name, persona_path=persona_path, agent=args.agent,
                llm=args.llm, seed=args.seed, max_steps=args.max_steps,
                eval_task_ids=eval_ids, notes=f"q3 arm={arm}")


    tier = args.feedback_tier

    if arm == "none":
        return SessionConfig(stream_task_ids=[], checkpoints=[0], memory_mode="none",
                             feedback_tier=tier, **base), NoOpUpdater()
    if arm == "oracle":
        return SessionConfig(stream_task_ids=[], checkpoints=[0], memory_mode="oracle",
                             oracle_verbosity=1, feedback_tier=tier,
                             **base), NoOpUpdater()

    stream = build_stream(pool, args.n_train, args.seed)
    if arm == "selfevolve_rewrite":
        return SessionConfig(stream_task_ids=stream, checkpoints=args.checkpoints,
                             memory_mode="updater", feedback_tier=tier,
                             **base), SelfEvolveUpdater(build_llm(args.llm),
                                                        max_memory_chars=args.max_memory_chars)
    if arm == "selfevolve_ace":
        return SessionConfig(stream_task_ids=stream, checkpoints=args.checkpoints,
                             memory_mode="updater", feedback_tier=tier,
                             **base), ACEStyleUpdater(build_llm(args.llm),
                                                      max_bullets=args.max_bullets)
    if arm == "external_corrective":
        return SessionConfig(stream_task_ids=stream, checkpoints=args.checkpoints,
                             memory_mode="updater", feedback_tier=tier,
                             **base), SelfEvolveUpdater(build_llm(args.llm),
                                                        max_memory_chars=args.max_memory_chars)


    if arm == "reflexion":
        return SessionConfig(stream_task_ids=stream, checkpoints=args.checkpoints,
                             memory_mode="updater", feedback_tier=tier,
                             **base), ReflexionUpdater(build_llm(args.llm),
                                                       window=args.reflexion_window)
    if arm == "tepa":
        return SessionConfig(stream_task_ids=stream, checkpoints=args.checkpoints,
                             memory_mode="updater", feedback_tier=tier,
                             **base), TepaUpdater(build_llm(args.llm))
    if arm == "trace":


        return SessionConfig(stream_task_ids=stream, checkpoints=args.checkpoints,
                             memory_mode="updater", feedback_tier=tier,
                             self_gate_attempts=args.trace_gate_attempts,
                             **base), TraceUpdater(build_llm(args.llm),
                                                   max_rules=args.max_bullets)
    raise ValueError(f"unknown arm: {arm}")


def _is_complete(run_name, arm, args) -> bool:
    """Does this arm already have every eval episode it should?

    Counts eval episodes per checkpoint rather than lines in the file: a run killed mid-write
    leaves a partial episodes.jsonl, and treating that as done would report a half-finished arm
    as a result. Same reasoning as run_exp1._arm_is_complete.
    """
    path = OUTPUTS_DIR / "sessions" / run_name / "episodes.jsonl"
    if not path.exists():
        return False
    try:
        rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    except json.JSONDecodeError:
        return False
    want_points = args.checkpoints if arm in EVOLVING_ARMS else [0]
    want_per_point = args.n_eval * args.rollouts
    for n in want_points:
        got = sum(1 for r in rows if r.get("phase") == "eval" and r.get("n") == n)
        if got < want_per_point:
            return False
    return True


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--llm", required=True)
    p.add_argument("--persona", default="q3_self_evolution")
    p.add_argument("--agent", default="fc", choices=["react", "fc", "oracle"])
    p.add_argument("--arms", nargs="+", default=["selfevolve_ace"], choices=sorted(EVOLVING_ARMS | {"none", "oracle"}))
    p.add_argument("--n-train", type=int, default=24)
    p.add_argument("--checkpoints", nargs="+", type=int, default=[0, 6, 12, 18, 24])
    p.add_argument("--n-eval", type=int, default=6)
    p.add_argument("--rollouts", type=int, default=3,
                   help="repeats of each eval task per checkpoint; the eval set is 6 tasks, so "
                        "this is what makes the per-rule denominator usable")
    p.add_argument("--max-steps", type=int, default=50)


    p.add_argument("--max-bullets", type=int, default=12)
    p.add_argument("--max-memory-chars", type=int, default=1200)


    p.add_argument("--reflexion-window", type=int, default=3)


    p.add_argument("--trace-gate-attempts", type=int, default=3)
    p.add_argument("--feedback-tier", default="instance",
                   choices=["instance", "corrective", "default", "vague_scoped", "aspect"],
                   help="feedback precision, ordered by information content: default (bare "
                        "dissatisfaction) < vague_scoped (names the artefact) < aspect (names which "
                        "aspect of it, not the target value) < "
                        "instance (names the rule via the checker detail) < corrective (hands "
                        "over the correction template)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--skip-done", action="store_true")
    p.add_argument("--tag", default="")
    args = p.parse_args()

    pool = json.loads(TASK_POOL_PATH.read_text())["pools"][args.persona]
    persona_path = str(PERSONA_DIR / f"{args.persona}.yaml")
    slug = args.llm.split(":")[-1].replace(".", "_")
    stem = f"q3{args.tag}_{slug}_{args.persona}_s{args.seed}"
    out = OUTPUTS_DIR / f"{stem}.json"

    index = {"llm": args.llm, "persona": args.persona, "seed": args.seed,
             "arms": {}, "args": vars(args)}
    if out.exists():
        try:
            index["arms"] = json.loads(out.read_text()).get("arms", {})
        except (OSError, json.JSONDecodeError) as exc:
            print(f"  !! could not read existing index ({exc}); starting fresh")

    print(f"persona={args.persona} eval={args.n_eval}x{args.rollouts} rollouts "
          f"stream={args.n_train} checkpoints={args.checkpoints}")
    for arm in args.arms:
        run_name = f"{stem}_{arm}"
        if args.skip_done and _is_complete(run_name, arm, args):
            print(f"\n##### {arm} -> already complete, skipping")
            continue
        print(f"\n##### {arm} -> {run_name}")
        config, updater = arm_config(arm, run_name, persona_path, pool, args)
        manifest = SessionDriver(config, updater).run()
        entry = {"run_name": run_name, "eval_points": manifest["eval_points"],
                 "llm_usage": manifest["llm_usage"], "updater_state": updater.state()}
        index["arms"][arm] = entry


        if hasattr(updater, "save"):
            updater.save(OUTPUTS_DIR / "sessions" / run_name / "updater_final.json")
        out.write_text(json.dumps(index, indent=1))
        print(f"  updater_state: {updater.state()}")

    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
