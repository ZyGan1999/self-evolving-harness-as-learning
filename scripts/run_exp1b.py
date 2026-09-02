"""Exp 1B (Q1, family B = statistical aggregation): arm runner.

The x-axis here is habit OBSERVATIONS, not AppWorld episodes: the habit stream is
synthetic micro-rounds (appworld_p/habit.py), so log length sweeps for free and the
only LLM cost is the frozen eval set. Every arm sees the identical stream; the arms
differ only in the computation applied to it (information parity, Q1_REDESIGN 1.2):

  baseline  empty memory                                  -> floor
  oracle    adoptable rule statement (names the dimension,
            never the bank)                               -> chance-level by design
  fulllog   statement + the complete raw log              -> context-class upper bound
                                                             (prediction: flat in n)
  external  statement + acceptance-rate counter argmax    -> control class (~1.0)

Usage:
  python scripts/run_exp1b.py --llm anthropic:claude-haiku-4-5-20251001 \
      --arms baseline oracle fulllog external --schedule 0 20 60 120 \
      --dominant "American Express" [--tier binary] [--seed 0] [--tag _pilot]
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from appworld_p.config import OUTPUTS_DIR, PERSONA_DIR  # noqa: E402
from appworld_p.driver import SessionConfig, SessionDriver  # noqa: E402
from appworld_p.habit import FAMILY_B_EVAL_TASKS, FAMILY_B_OPTIONS  # noqa: E402
from appworld_p.updaters import NoOpUpdater  # noqa: E402

ARMS = ("baseline", "oracle", "fulllog", "external", "gate")


def arm_config(arm: str, run_name: str, persona_path: str,
               args: argparse.Namespace) -> SessionConfig:
    # baseline/oracle carry no stream-dependent state: evaluating them once (at the
    # largest n) is sufficient and saves episodes
    schedule = args.schedule if arm in ("fulllog", "external") else [args.schedule[-1]]
    base = dict(run_name=run_name, persona_path=persona_path, agent=args.agent,
                llm=args.llm, seed=args.seed, max_steps=args.max_steps,
                eval_task_ids=list(FAMILY_B_EVAL_TASKS)[: args.n_eval],
                stream_task_ids=[], checkpoints=[],
                habit_dominant=args.dominant, habit_tier=args.tier,
                habit_noisy=args.noisy, habit_noise_per_round=args.noise_density,
                habit_schedule=schedule,
                notes=f"exp1b arm={arm} tier={args.tier} dominant={args.dominant}")
    if arm == "baseline":
        return SessionConfig(memory_mode="none", **base)
    if arm == "oracle":
        return SessionConfig(memory_mode="oracle", oracle_verbosity=1, **base)
    if arm == "fulllog":
        return SessionConfig(memory_mode="fulllog", oracle_verbosity=1, **base)
    if arm == "external":
        return SessionConfig(memory_mode="oracle", oracle_verbosity=1,
                             inject_stats=True, **base)
    if arm == "gate":
        # reject-only control harness, run through the NON-LEAKING channel: this rule's ordinary
        # detail names the habitual bank, so verifier_detail here would measure label-copying.
        # Exclusions accumulate because each retry gets a fresh world and context.
        return SessionConfig(memory_mode="oracle", oracle_verbosity=1,
                             verifier_attempts=args.attempts, verifier_nonleaking=True,
                             verifier_accumulate=True, **base)
    raise ValueError(f"unknown arm: {arm}")


def summarize(run_name: str) -> dict:
    """Family-B readout: per-n share of episodes whose FIRST card is the habit."""
    path = OUTPUTS_DIR / "sessions" / run_name / "episodes.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    by_n: dict[str, dict] = {}
    for r in rows:
        slot = by_n.setdefault(str(r["n"]), {"episodes": 0, "hits": 0, "tgc": 0,
                                             "no_card": 0, "cards": {}})
        rule = r["rules"].get("usual_card", {})
        slot["episodes"] += 1
        slot["tgc"] += int(bool(r["tgc"]))
        slot["hits"] += int(rule.get("satisfied") is True and rule.get("applicable"))
        card = r.get("first_card")
        if card is None:
            slot["no_card"] += 1
        else:
            slot["cards"][card] = slot["cards"].get(card, 0) + 1
    for slot in by_n.values():
        slot["hit_rate"] = round(slot["hits"] / slot["episodes"], 3) if slot["episodes"] else None
    return by_n


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm", required=True)
    parser.add_argument("--persona", default="p3_fc")
    parser.add_argument("--agent", default="fc", choices=["fc", "react"],
                        help="react = the actuator-flip control slice")
    parser.add_argument("--arms", nargs="+", default=list(ARMS))
    parser.add_argument("--schedule", nargs="+", type=int, default=[0, 20, 60, 120],
                        help="cumulative habit observations to evaluate at")
    parser.add_argument("--dominant", default=FAMILY_B_OPTIONS[0], choices=FAMILY_B_OPTIONS,
                        help="bank carrying most of pi_u (rotate = brand-prior control)")
    parser.add_argument("--attempts", type=int, default=3,
                        help="gate arm only: retries per episode. Elimination over the world's "
                             "4-5 cards is not guaranteed at 3, but the ceiling is reported "
                             "alongside the measurement (scripts/gate_elimination_ceiling.py)")
    parser.add_argument("--tier", default="binary", choices=["binary", "corrective"],
                        help="feedback information q: binary = out regime")
    parser.add_argument("--noisy", action="store_true",
                        help="dilute the log with unrelated interactions")
    parser.add_argument("--noise-density", type=float, default=1.0,
                        help="mean unrelated lines per habit round (with --noisy)")
    parser.add_argument("--n-eval", type=int, default=6)
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tag", default="")
    args = parser.parse_args()

    persona_path = str(PERSONA_DIR / f"{args.persona}.yaml")
    model_slug = args.llm.split(":")[-1].replace(".", "_")
    noise = f"_d{args.noise_density:g}" if args.noisy else ""
    slug = (f"exp1b{args.tag}_{model_slug}_{args.dominant.split()[0].lower()}"
            f"_{args.tier}{noise}_s{args.seed}")

    index = {"llm": args.llm, "persona": args.persona, "args": vars(args), "arms": {}}
    for arm in args.arms:
        run_name = f"{slug}_{arm}"
        print(f"\n##### arm {arm} -> {run_name}", flush=True)
        config = arm_config(arm, run_name, persona_path, args)
        manifest = SessionDriver(config, NoOpUpdater()).run()
        index["arms"][arm] = {"run_name": run_name, "habit": manifest["habit"],
                              "by_n": summarize(run_name),
                              "llm_usage": manifest["llm_usage"]}
        print(f"  {arm} by_n: " + json.dumps(index['arms'][arm]['by_n']), flush=True)

    out = OUTPUTS_DIR / f"{slug}.json"
    out.write_text(json.dumps(index, indent=1))

    print(f"\n=== exp1b: target={args.dominant} tier={args.tier} seed={args.seed} ===")
    ns = [str(n) for n in args.schedule]
    print(f"{'arm':10s} " + " ".join(f"{'n=' + n:>9s}" for n in ns))
    for arm, res in index["arms"].items():
        cells = [f"{res['by_n'][n]['hit_rate']:>9.2f}" if n in res["by_n"] else f"{'-':>9s}"
                 for n in ns]
        print(f"{arm:10s} " + " ".join(cells))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
