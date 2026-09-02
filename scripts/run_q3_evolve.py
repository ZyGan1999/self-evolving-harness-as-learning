"""Q3: does self-evolution plateau above the oracle?

Runs the on-policy self-evolution loop the driver already supports but no script has used yet:
every train episode is answered with the CURRENT memory, the learner then rewrites that memory
from the user's complaint, and the next episode sees the new one. Q1's `learned` arm is not this
-- its candidate bank is built from `rule.oracle_text` (updaters.py), so it SELECTS among
pre-written perfect statements instead of generating its own. That difference is the measurement:

    none -> selfevolve -> bandit(oracle bank) -> oracle
            cost of WRITING the hypothesis / cost of PICKING it

Arms:
  none                 empty memory. Confirms the preferences are implicit.
  oracle               oracle_text. The reachability reference; the plateau is measured against
                       THIS, not against 0, because a rule whose oracle arm sits above 0 is
                       Q1's approximation error and not Q3's optimisation error.
  selfevolve_rewrite   community-standard recipe: the learner rewrites the whole memory each
                       time (SelfEvolveUpdater). This is the one ACE documents collapsing.
  selfevolve_ace       ACE recipe: itemised bullets, delta updates, deterministic merge, dedup,
                       fixed capacity (ace_updater.py). Present so a plateau cannot be dismissed
                       as an artefact of a strawman updater (pre-registered risk 4).
  external_corrective  same rewrite updater, but fed the corrective tier instead of instance.
                       Diagnostic: corrective hands over correction_template, which for
                       payment_note_category contains the literal '[personal]' the instance
                       complaint never carries. If a rule is stuck under instance and moves under
                       corrective, the ceiling is in the FEEDBACK CHANNEL; if it is stuck under
                       both, it is in the learner.

  Published-recipe baselines (appworld_p/baseline_updaters.py). ACE plus the naive rewrite occupy
  two cells of a 2x2 over capacity control x update granularity; these fill the rest, so a
  plateau cannot be read as a property of one recipe's bookkeeping:
  reflexion            Reflexion (arXiv:2303.11366): append-only reflection buffer, hard FIFO
                       window W=3. Its ceiling is PREDICTABLE -- a W-line window cannot hold more
                       than W preferences -- so it separates a capacity ceiling from a channel one.
  cheatsheet           Dynamic Cheatsheet (arXiv:2504.07952): curator regenerates the whole sheet
                       each round with explicit preserve/consolidate instructions, no hard cap.
                       The intermediate point between ACE's targeted edit and naive rewrite.
  amem                 A-MEM (arXiv:2502.12110): unbounded note set, never evicts, and rewrites a
                       NEIGHBOUR's framing rather than its content. If the plateau were caused by
                       ACE's fixed capacity, this arm should escape it.

Usage:
  python scripts/run_q3_evolve.py --llm anthropic:claude-haiku-4-5-20251001 \
      --persona p9_q3 --agent fc --arms selfevolve_rewrite \
      --n-train 12 --checkpoints 0 3 6 12 --n-eval 6 --rollouts 2 --seed 0 --tag _pilot
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from appworld_p.ace_updater import ACEStyleUpdater  # noqa: E402
from appworld_p.baseline_updaters import (AMemUpdater, DynamicCheatsheetUpdater,  # noqa: E402
                                          ReflexionUpdater, TepaUpdater, TraceUpdater)
from appworld_p.config import OUTPUTS_DIR, PERSONA_DIR, TASK_POOL_PATH  # noqa: E402
from appworld_p.driver import SessionConfig, SessionDriver  # noqa: E402
from appworld_p.llm import build_llm  # noqa: E402
from appworld_p.streams import build_stream  # noqa: E402
from appworld_p.updaters import NoOpUpdater, SelfEvolveUpdater  # noqa: E402

# Arms whose memory changes along the stream, so they are evaluated at every checkpoint. The
# static arms are flat in n by construction and only need n=0.
EVOLVING_ARMS = {"selfevolve_rewrite", "selfevolve_ace", "external_corrective",
                 "reflexion", "cheatsheet", "amem", "tepa", "trace"}


def arm_config(arm, run_name, persona_path, pool, args):
    """(SessionConfig, updater) for one arm."""
    eval_ids = pool["eval_task_ids"][: args.n_eval]
    # rollouts: the eval set is only 6 tasks, so each is repeated to get a usable denominator.
    # Repeating the id list is what the driver understands; it opens a fresh AppWorld per entry.
    eval_ids = [t for t in eval_ids for _ in range(args.rollouts)]
    base = dict(run_name=run_name, persona_path=persona_path, agent=args.agent,
                llm=args.llm, seed=args.seed, max_steps=args.max_steps,
                eval_task_ids=eval_ids, notes=f"q3 arm={arm}")

    # Feedback tier: overridable via --feedback-tier. The external_corrective arm used to hardcode
    # 'corrective', but now we let all arms inherit the command-line default so a single run can
    # test different tiers on the same updater. The arm name stays 'external_corrective' for
    # backward compatibility with analysis scripts that expect it.
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
    # Published-recipe baselines. All three take the same (episode, feedback) the other arms take
    # and need no reward, no labels and no extra rollouts, which is what made them admissible
    # (see appworld_p/baseline_updaters.py for the ones that were ruled out and why).
    if arm == "reflexion":
        return SessionConfig(stream_task_ids=stream, checkpoints=args.checkpoints,
                             memory_mode="updater", feedback_tier=tier,
                             **base), ReflexionUpdater(build_llm(args.llm),
                                                       window=args.reflexion_window)
    if arm == "cheatsheet":
        return SessionConfig(stream_task_ids=stream, checkpoints=args.checkpoints,
                             memory_mode="updater", feedback_tier=tier,
                             **base), DynamicCheatsheetUpdater(
                                 build_llm(args.llm), max_memory_chars=args.max_memory_chars)
    if arm == "amem":
        return SessionConfig(stream_task_ids=stream, checkpoints=args.checkpoints,
                             memory_mode="updater", feedback_tier=tier,
                             **base), AMemUpdater(build_llm(args.llm), top_k=args.amem_top_k)
    if arm == "tepa":
        return SessionConfig(stream_task_ids=stream, checkpoints=args.checkpoints,
                             memory_mode="updater", feedback_tier=tier,
                             **base), TepaUpdater(build_llm(args.llm))
    if arm == "trace":
        # self_gate_attempts, NOT verifier_attempts: the gate runs the updater's own compiled
        # checks. Using the persona checkers would leak ground truth and make this control-class.
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
    p.add_argument("--persona", default="p9_q3")
    p.add_argument("--agent", default="fc", choices=["react", "fc", "oracle"])
    p.add_argument("--arms", nargs="+", default=["selfevolve_rewrite"])
    p.add_argument("--n-train", type=int, default=24)
    p.add_argument("--checkpoints", nargs="+", type=int, default=[0, 2, 4, 6, 9, 12, 18, 24])
    p.add_argument("--n-eval", type=int, default=6)
    p.add_argument("--rollouts", type=int, default=3,
                   help="repeats of each eval task per checkpoint; the eval set is 6 tasks, so "
                        "this is what makes the per-rule denominator usable")
    p.add_argument("--max-steps", type=int, default=50)
    # Capacity caps, both aimed at the same hazard: Q2 measured violation rising with block
    # length, and line ORDER moving it 1.89x more than length. A memory free to grow would
    # confound Q3's optimisation error with those effects.
    p.add_argument("--max-bullets", type=int, default=12)
    p.add_argument("--max-memory-chars", type=int, default=1200)
    # Reflexion's buffer width. The paper uses 1-3 (ALFWorld/HotpotQA 3, programming 1); 3 is the
    # widest it reports and therefore the most favourable setting for the baseline, which is the
    # one to run when the arm is expected to hit a capacity ceiling.
    p.add_argument("--reflexion-window", type=int, default=3)
    # A-MEM's link fan-out. The paper retrieves top-k=10 for answering; k here governs how many
    # neighbours a new note may recontextualise, and each one costs an LLM call.
    p.add_argument("--amem-top-k", type=int, default=3)
    # TRACE's compiled checks must pass before the agent may finish. 3 matches the gate budget
    # Q1 used, so a rejection loop costs the same there as here.
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
        # The bullet-level history is the audit trail for a plateau: it separates "the learner
        # proposed nothing" from "it proposed something unparseable" from "it kept rewriting the
        # same wrong scope", which the violation curve alone cannot distinguish.
        if hasattr(updater, "save"):
            updater.save(OUTPUTS_DIR / "sessions" / run_name / "updater_final.json")
        out.write_text(json.dumps(index, indent=1))
        print(f"  updater_state: {updater.state()}")

    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
