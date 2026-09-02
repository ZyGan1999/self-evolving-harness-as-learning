"""M1 / Block 0: per-rule calibration for one model.

For every rule of the given personas, run trigger tasks under three fixed contexts:
  none    — empty memory                      -> base compliance rate
  oracle  — the rule's clearest NL statement  -> oracle compliance (in/out-of-support test)
  anti    — the opposite assertion            -> eta (fidelity: does f follow memory, even
                                                 when it contradicts base behavior?)

Classification (per model): oracle compliance >= 0.95 -> in-support candidate confirmed;
oracle compliance plateaus low (< 0.70) -> out-of-support for this model.
eta proxy = anti-arm violation rate of the original rule (following the anti assertion
means violating the rule).

Usage:
  python scripts/run_calibration.py --llm anthropic:claude-haiku-4-5-20251001 \
      --personas p1_style p2_mixed --tasks-per-rule 3 [--rules sms_signoff ...] [--arms none oracle anti]
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from appworld_p.config import OUTPUTS_DIR, PERSONA_DIR, TASK_POOL_PATH  # noqa: E402
from appworld_p.driver import SessionConfig, SessionDriver  # noqa: E402
from appworld_p.persona import Persona  # noqa: E402
from appworld_p.rules import RULE_REGISTRY  # noqa: E402
from appworld_p.updaters import NoOpUpdater  # noqa: E402


def tasks_triggering(rule_name: str, metadata: list[dict], max_difficulty: int = 2) -> list[str]:
    rule = RULE_REGISTRY[rule_name]()
    apis = rule.trigger_apis()
    hits = [t["task_id"] for t in metadata
            if (t["difficulty"] or 99) <= max_difficulty
            and apis & set(t.get("required_apis", []))]
    # Round-robin across task families (task_id prefix): the first k hits are k
    # variants of ONE generator template, so a family-specific quirk (TGC-hard
    # task, instruction/preference conflict) contaminates the whole measurement.
    by_family: dict[str, list[str]] = {}
    for tid in hits:
        by_family.setdefault(tid.split("_")[0], []).append(tid)
    families = sorted(by_family)
    ordered = []
    while any(by_family[f] for f in families):
        for f in families:
            if by_family[f]:
                ordered.append(by_family[f].pop(0))
    return ordered


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm", required=True)
    parser.add_argument("--agent", default="react", choices=["react", "fc"],
                        help="actuator: react = code agent, fc = constrained "
                             "function-call agent (Q1 redesign)")
    parser.add_argument("--personas", nargs="+", default=["p1_style", "p2_mixed"])
    parser.add_argument("--rules", nargs="+", default=None,
                        help="restrict to specific rules (default: all persona rules)")
    parser.add_argument("--arms", nargs="+", default=["none", "oracle", "anti"])
    parser.add_argument("--tasks-per-rule", type=int, default=3)
    # Rules defined over the whole transaction history need enough steps to page through it:
    # at 30 the agent spends every step paginating and never pays, which reads as
    # applicable=False and confounds "cannot sum" with "ran out of budget".
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--tag", default="")
    args = parser.parse_args()

    metadata = json.loads(TASK_POOL_PATH.read_text())["task_metadata"]
    model_slug = args.llm.split(":")[-1].replace(".", "_")

    rule_names: list[str] = []
    rule_to_persona: dict[str, str] = {}
    for pname in args.personas:
        persona = Persona.load(PERSONA_DIR / f"{pname}.yaml")
        for r in persona.rules:
            if r.name not in rule_to_persona:
                rule_to_persona[r.name] = pname
                rule_names.append(r.name)
    if args.rules:
        unknown = [r for r in args.rules if r not in RULE_REGISTRY]
        if unknown:
            parser.error(f"unknown rule(s): {unknown}")
        # a requested rule need not belong to the persona -- calibration measures the
        # rule, not the persona. Silently intersecting used to drop such rules with no
        # warning (sms_char_checksum vanished from the famA run this way).
        for name in args.rules:
            if name not in rule_to_persona:
                rule_to_persona[name] = args.personas[0]
                rule_names.append(name)
                print(f"note: {name} is not in {args.personas[0]}; calibrating it anyway")
        rule_names = [r for r in rule_names if r in args.rules]

    results: dict[str, dict] = {}
    for rule_name in rule_names:
        rule = RULE_REGISTRY[rule_name]()
        task_ids = tasks_triggering(rule_name, metadata)[: args.tasks_per_rule]
        if not task_ids:
            print(f"!! {rule_name}: no trigger tasks, skipped")
            continue
        results[rule_name] = {"pool": rule.pool, "n_tasks": len(task_ids), "arms": {}}
        for arm in args.arms:
            if arm == "none":
                memory = ""
            elif arm == "oracle":
                memory = "My preference: " + rule.oracle_text
            elif arm == "anti":
                if not rule.negation_text:
                    continue
                memory = "My preference: " + rule.negation_text
            else:
                raise ValueError(arm)
            run_name = f"calib{args.tag}_{model_slug}_{rule_name}_{arm}"
            config = SessionConfig(
                run_name=run_name,
                persona_path=str(PERSONA_DIR / f"{rule_to_persona[rule_name]}.yaml"),
                stream_task_ids=[], eval_task_ids=task_ids, checkpoints=[0],
                agent=args.agent, llm=args.llm, max_steps=args.max_steps,
                memory_mode="fixed", fixed_memory=memory,
                extra_rules=(rule_name,),   # score it even if the persona lacks it
                notes=f"calibration {rule_name} {arm}",
            )
            manifest = SessionDriver(config, NoOpUpdater()).run()
            rows = [json.loads(line) for line in
                    (OUTPUTS_DIR / "sessions" / run_name / "episodes.jsonl").read_text().splitlines()]
            applicable = sum(1 for r in rows if r["rules"].get(rule_name, {}).get("applicable"))
            violated = sum(1 for r in rows if r["rules"].get(rule_name, {}).get("applicable")
                           and r["rules"][rule_name]["satisfied"] is False)
            # satisfactions that hold only because the precondition is empty
            # (e.g. no card history yet) carry no in/out evidence
            vacuous = sum(1 for r in rows if r["rules"].get(rule_name, {}).get("applicable")
                          and r["rules"][rule_name]["satisfied"]
                          and "vacuous" in str(r["rules"][rule_name].get("detail", "")))
            tgc = sum(1 for r in rows if r["tgc"])
            tgc_applicable = sum(1 for r in rows if r["tgc"]
                                 and r["rules"].get(rule_name, {}).get("applicable"))
            tgc_violated = sum(1 for r in rows if r["tgc"]
                               and r["rules"].get(rule_name, {}).get("applicable")
                               and r["rules"][rule_name]["satisfied"] is False)
            results[rule_name]["arms"][arm] = {
                "episodes": len(rows), "applicable": applicable, "violated": violated,
                "vacuous": vacuous,
                # all-vacuous means the precondition never fired (e.g. usual_card with no
                # habit stream): 1.00 there is "never tested", not "always complied", and
                # reporting it as compliance would fake an in-support verdict
                "compliance": (round(1 - violated / applicable, 3)
                               if applicable and vacuous < applicable else None),
                "compliance_note": ("all satisfactions vacuous — rule never actually tested"
                                    if applicable and vacuous >= applicable else None),
                "compliance_tgc": (round(1 - tgc_violated / tgc_applicable, 3)
                                   if tgc_applicable else None),
                "tgc": tgc,
            }
            vac = f" vacuous={vacuous}" if vacuous else ""
            print(f"  {rule_name:24s} {arm:6s} applicable={applicable} "
                  f"violated={violated}{vac} tgc={tgc}/{len(rows)}")

    out_path = OUTPUTS_DIR / f"calibration_{model_slug}{args.tag}.json"
    out_path.write_text(json.dumps({"llm": args.llm, "results": results}, indent=1))

    print(f"\n=== calibration table ({args.llm}) ===")
    print(f"{'rule':26s} {'pool':4s} {'base':>6s} {'oracle':>7s} {'orc|tgc':>8s} {'anti(eta)':>9s}  class")
    for name, res in results.items():
        arms = res["arms"]
        base = arms.get("none", {}).get("compliance")
        orc_arm = arms.get("oracle", {})
        orc = orc_arm.get("compliance")
        orc_tgc = orc_arm.get("compliance_tgc")
        anti = arms.get("anti", {}).get("compliance")
        eta = None if anti is None else round(1 - anti, 3)  # following anti = violating rule
        orc_evidence = (orc_arm.get("applicable") or 0) - (orc_arm.get("vacuous") or 0)
        if orc is None or orc_evidence == 0:
            cls = "UNMEASURED (vacuous)" if orc is not None else "borderline"
        elif orc >= 0.95:
            cls = "in-support"
        elif orc < 0.70:
            cls = "OUT-of-support"
        else:
            cls = "borderline"
        fmt = lambda x: "  n/a" if x is None else f"{x:5.2f}"
        print(f"{name:26s} {res['pool']:4s} {fmt(base):>6s} {fmt(orc):>7s} {fmt(orc_tgc):>8s} "
              f"{fmt(eta):>9s}  {cls}")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
