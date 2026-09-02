"""Q2 R/D/N sweep: at equal L, vary only what memory is MADE of.

R  every line paraphrases one of the d scored preferences        relevance 1, flat expected
D  d on-topic lines + correct lines about never-exercised prefs  relevance d/L, rise expected
N  a constant fraction q of lines assert the reverse             relevance 1-q, flat expected

R reproduces the earlier null (redundancy out-votes noise). N is the negative control for
"noise alone does not scale". D is the hypothesis: memory grows by accumulating correct lines
that do not apply here, and the share that does apply falls as 1/L.

Usage:
  python scripts/run_exp2_arms.py --llm anthropic:claude-haiku-4-5-20251001 \
      --arms R D N --l-values 5 20 60 150 --seed 1 --tag _v1 --donor-tag _d1
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from appworld_p.arms import build_arm_memory, filter_donor  # noqa: E402
from appworld_p.conflict import ConflictGraph  # noqa: E402
from appworld_p.rules.base import RULE_REGISTRY  # noqa: E402
from appworld_p.config import OUTPUTS_DIR, PERSONA_DIR, TASK_POOL_PATH  # noqa: E402
from appworld_p.driver import SessionConfig, SessionDriver  # noqa: E402
from appworld_p.conflict import Assertion as _CA  # noqa: E402
from appworld_p.summarize import Assertion  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_inducibility import read_rows, score  # noqa: E402


def _A(rule: str):
    """A rule asserted in the direction memory would write it: positively.

    Assertion.direction is the STRING '+' or '-' -- ConflictGraph keys its acceptance table by
    those literals. This passed the bool True until 2026-08, which is not a key in that table,
    so graph.conflicts() returned False for every pair and the donor-collision guard below
    could never fire. Verified after the fix: conflicts(venmo_private, venmo_public_feed) is
    True with '+' and False with True.
    """
    return _CA(rule, "+")
from appworld_p.updaters import NoOpUpdater  # noqa: E402


def load_pool(tag: str, model_slug: str, persona: str, seed: int) -> list:
    p = OUTPUTS_DIR / f"exp2pool{tag}_{model_slug}_{persona}_s{seed}.json"
    return [Assertion(**a) for a in json.loads(p.read_text())["assertions"]]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", required=True)
    ap.add_argument("--persona", default="p6_q2")
    ap.add_argument("--donor", default="p7_offtopic")
    ap.add_argument("--arms", nargs="+", default=["R", "D", "N"])
    # Spans BOTH sides of d. Below d the three arms are byte-identical (nothing is left over
    # to differ in), so the falling branch is shared and is driven by coverage alone; above d
    # they diverge. A U therefore has to be read across d, not within L > d: the two scored
    # preferences with headroom are already at zero violation at L = d, so above it there is
    # only room to rise.
    ap.add_argument("--l-values", nargs="+", type=int, default=[1, 2, 3, 5, 20, 60, 150])
    ap.add_argument("--q", type=float, default=0.3, help="N arm's constant noise fraction")
    ap.add_argument("--pick", default="consensus", choices=["consensus", "earliest"])
    ap.add_argument("--cap-distinct", type=int, default=None,
                    help="distinct texts per arm; defaults to the admitted donor pool's count "
                         "so D and R differ in relevance only, not in repetitiveness")
    ap.add_argument("--skip-below-d", action="store_true",
                    help="run L<=d for the first arm only: below d the arms are byte-identical")
    ap.add_argument("--n-eval", type=int, default=6)
    ap.add_argument("--max-steps", type=int, default=50)
    ap.add_argument("--agent", default="fc", choices=["fc", "react"])
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--pool-seeds", nargs="+", type=int, default=[])
    ap.add_argument("--rollouts", type=int, default=1,
                    help="independent rollouts per eval task (temperature 0.7). Multiplies "
                         "cost by K; shrinks per-task binomial noise, does not add coverage.")
    ap.add_argument("--only-eval", nargs="+", default=[],
                    help="restrict the eval set to these task ids (must already be in the "
                         "pool's eval set). Use to spend rollouts only on tasks that carry "
                         "the rule under test instead of on all 6.")
    ap.add_argument("--tag", default="_v1", help="scored pool tag")
    ap.add_argument("--donor-tag", default="_d1")
    ap.add_argument("--donor-seed", type=int, default=1,
                    help="which donor collection to use; held fixed across --seed so that "
                         "replication and filler-robustness stay separate questions")
    ap.add_argument("--out-suffix", default="")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    slug = args.llm.split(":")[-1].replace(".", "_")
    pool_seeds = args.pool_seeds or [args.seed]
    scored = [a for s in pool_seeds for a in load_pool(args.tag, slug, args.persona, s)]
    # Donor content is held FIXED across eval seeds. --seed varies line order and LLM
    # sampling; letting it also swap the filler would confound "does this replicate" with
    # "does it hold for other filler", two questions that need separate answers.
    raw_donor = load_pool(args.donor_tag, slug, args.donor, args.donor_seed)
    pools = json.loads(TASK_POOL_PATH.read_text())["pools"]
    pool = pools[args.persona]
    eval_ids = pool["eval_task_ids"][: args.n_eval]
    if args.only_eval:
        missing = set(args.only_eval) - set(eval_ids)
        if missing:
            sys.exit(f"--only-eval ids not in this pool's eval set: {sorted(missing)}")
        eval_ids = [t for t in eval_ids if t in args.only_eval]
    # Repeat each eval id K times in the SAME list. The driver enumerates this list and folds
    # the position j into experiment_name, so each repeat opens its own AppWorld and is a fresh
    # rollout, not a cache hit on the first one. Rollouts estimate the per-task compliance
    # probability at temperature 0.7; they do not widen task coverage, so they answer "is 0.17
    # noise" and not "does this hold beyond these tasks". Analysis must cluster on task.
    eval_ids = [t for t in eval_ids for _ in range(args.rollouts)]
    d = len(pool["eval_rule_cover"])
    # The scored rules, and how many eval episodes each one can apply to. Carried into the
    # output so the per-rule rates have their denominators attached: the eval split holds only
    # 6 tasks, so a rule covered twice moves in steps of 0.5 and cannot support a rate read to
    # two decimals.
    persona_rules = list(pool["eval_rule_cover"])
    print(f"per-rule eval coverage: {pool['eval_rule_cover']}")

    # Non-collision and donor admission are decided by executing the rules' checkers, not by a
    # list maintained here; the graph also reports which rules it cannot test, and lines from
    # those are dropped. Printed so the run log carries the verdict it relied on.
    names = sorted({a.rule_name for a in scored} | {a.rule_name for a in raw_donor})
    graph = ConflictGraph({n: RULE_REGISTRY[n]() for n in names})
    donor, dropped = filter_donor(raw_donor, set(graph.testable()))
    print(f"conflict graph untestable: {graph.untestable()}")
    print(f"donor {len(raw_donor)} -> {len(donor)} admitted "
          f"({len({a.text for a in donor})} distinct), {len(dropped)} dropped")
    for a in dropped:
        print(f"  dropped [{a.rule_name}] {a.text[:88]}")
    collide = [(dn, sn) for dn in {a.rule_name for a in donor} for sn in {a.rule_name for a in scored}
               if dn in graph.testable() and sn in graph.testable()
               and graph.conflicts(_A(dn), _A(sn))]
    print(f"donor x scored mechanical conflicts: {collide or 'none'}")
    if collide:
        sys.exit("donor collides with a scored preference: that is interference, not dilution")
    cap = args.cap_distinct if args.cap_distinct is not None else len({a.text for a in donor})
    print(f"scored pool {len(scored)} (seeds {pool_seeds}), d={d}, "
          f"{len(eval_ids)} eval tasks, cap_distinct={cap}")

    grid = []
    for ai, arm in enumerate(args.arms):
        for L in args.l_values:
            if args.skip_below_d and ai > 0 and L <= d:
                continue                          # identical to the first arm's cell by design
            memory, diag = build_arm_memory(arm, scored, donor, d, L, pick=args.pick,
                                            q=args.q, seed=args.seed, cap_distinct=cap)
            print(f"\n##### {arm} L={L} | on={diag['on_topic']} off={diag['off_topic']} "
                  f"r={diag['relevance']} chars={len(memory)} "
                  f"distinct_off={diag['distinct_off']}", flush=True)
            if args.dry_run:
                grid.append({**diag, "memory_chars": len(memory)})
                continue
            run = (f"exp2arms{args.tag}{args.out_suffix}_{slug}_{args.persona}"
                   f"_{arm}_L{L}_s{args.seed}")
            cfg = SessionConfig(
                run_name=run, persona_path=str(PERSONA_DIR / f"{args.persona}.yaml"),
                stream_task_ids=[], eval_task_ids=eval_ids, checkpoints=[0],
                agent=args.agent, llm=args.llm, seed=args.seed,
                max_steps=args.max_steps, memory_mode="fixed", fixed_memory=memory,
                notes=f"exp2 arms arm={arm} L={L} q={args.q} r={diag['relevance']}",
            )
            man = SessionDriver(cfg, NoOpUpdater()).run()
            # Per-rule breakdown, not just the aggregate: three of p6_q2's five rules are
            # reachable from the ORACLE text but not from the model's own induced text, and
            # they sit near 1.0 in every cell. They drag the aggregate toward a flat line and
            # can hide which rule a rise actually comes from. The applicable DENOMINATOR also
            # drifts cell to cell (a task that fails early never reaches the scored action),
            # so counts are carried alongside the rate.
            per_rule = {r: score(read_rows(run), r) for r in persona_rules}
            grid.append({**diag, "memory_chars": len(memory),
                         **man["eval_points"]["0"], "per_rule": per_rule})
            out = OUTPUTS_DIR / (f"exp2armsgrid{args.tag}{args.out_suffix}_{slug}"
                                 f"_{args.persona}_s{args.seed}.json")
            out.write_text(json.dumps({"args": vars(args), "grid": grid}, indent=1))
    if not args.dry_run:
        print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
