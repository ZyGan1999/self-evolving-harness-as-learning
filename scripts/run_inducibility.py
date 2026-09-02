"""Per-rule inducibility screen: can the MODEL'S OWN induced assertion fix the rule?

This is the screening criterion for building a Q2v2 persona with larger d. A rule qualifies
iff the model's self-induced memory (from instance-tier complaints) is sufficient to drive
agent compliance — as opposed to requiring oracle_text with specifics the complaint never
carried (e.g. which initials in round brackets, exact category tag '[personal]').

For each rule:
  phase A: run trigger tasks with EMPTY memory + instance feedback + AssertionCollector
           -> records base violation rate, collects induced assertion text
  phase B: re-run the SAME tasks with the induced assertion as FIXED memory
           -> records learned violation rate

Screening criteria:
  - base violation >= 0.6:         rule has headroom for memory to help
  - learned violation <= 0.1:      model's own text actually fixes it
  - testable in ConflictGraph:     conflicts with chosen persona are mechanically decided
  - >= 2 trigger task families:    eval set will have coverage

Usage:
  python scripts/run_inducibility.py --llm anthropic:claude-haiku-4-5-20251001 \
      --rules sms_signoff payment_note_category ... [--tag _screen1] [--tasks-per-rule 3]

Output: outputs/inducibility_{model_slug}{tag}.json
  {"rule_name": {"base_viol": 0.67, "learned_viol": 0.0, "induced_text": "...", ...}}
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
from appworld_p.rules import RULE_REGISTRY  # noqa: E402
from appworld_p.summarize import AssertionCollector  # noqa: E402
from appworld_p.updaters import NoOpUpdater  # noqa: E402


def tasks_triggering(rule_name: str, metadata: list[dict], max_difficulty: int = 2) -> list[str]:
    """Reused from run_calibration.py."""
    rule = RULE_REGISTRY[rule_name]()
    apis = rule.trigger_apis()
    hits = [t["task_id"] for t in metadata
            if (t["difficulty"] or 99) <= max_difficulty
            and apis & set(t.get("required_apis", []))]
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


def read_rows(run_name: str) -> list[dict]:
    path = OUTPUTS_DIR / "sessions" / run_name / "episodes.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines()]


def score(rows: list[dict], rule_name: str) -> dict:
    """Violation rate over episodes where the rule actually applied, non-vacuously."""
    applicable = [r for r in rows if r["rules"].get(rule_name, {}).get("applicable")]
    vacuous = [r for r in applicable
               if r["rules"][rule_name]["satisfied"]
               and "vacuous" in str(r["rules"][rule_name].get("detail", ""))]
    violated = [r for r in applicable if r["rules"][rule_name]["satisfied"] is False]
    n = len(applicable)
    return {
        "episodes": len(rows), "applicable": n, "violated": len(violated),
        "vacuous": len(vacuous),
        # all-vacuous means the precondition never fired: 0.00 violation there is
        # "never tested", not "complied", and must not read as a pass
        "viol_rate": (round(len(violated) / n, 3) if n and len(vacuous) < n else None),
        "note": ("all satisfactions vacuous — rule never tested"
                 if n and len(vacuous) >= n else None),
        "details": [r["rules"][rule_name].get("detail", "") for r in violated][:3],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm", required=True)
    parser.add_argument("--agent", default="fc", choices=["fc", "react"])
    parser.add_argument("--rules", nargs="+", required=True)
    parser.add_argument("--tasks-per-rule", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--tag", default="")
    parser.add_argument("--learner-llm", default="",
                        help="model that induces assertions; same as --llm by default")
    args = parser.parse_args()

    unknown = [r for r in args.rules if r not in RULE_REGISTRY]
    if unknown:
        parser.error(f"unknown rule(s): {unknown}")

    metadata = json.loads(TASK_POOL_PATH.read_text())["task_metadata"]
    model_slug = args.llm.split(":")[-1].replace(".", "_")
    persona_path = str(PERSONA_DIR / "p0_solo.yaml")
    learner = build_llm(args.learner_llm or args.llm)

    results: dict[str, dict] = {}
    for rule_name in args.rules:
        task_ids = tasks_triggering(rule_name, metadata)[: args.tasks_per_rule]
        if not task_ids:
            print(f"!! {rule_name}: no trigger tasks, skipped")
            continue
        entry: dict = {"n_tasks": len(task_ids), "task_ids": task_ids,
                       "families": sorted({t.split("_")[0] for t in task_ids})}

        # --- phase A: empty memory, instance feedback, collect induced assertions ---
        run_a = f"induc{args.tag}_{model_slug}_{rule_name}_A"
        collector = AssertionCollector(learner)
        SessionDriver(SessionConfig(
            run_name=run_a, persona_path=persona_path,
            stream_task_ids=task_ids, eval_task_ids=[], checkpoints=[],
            agent=args.agent, llm=args.llm, max_steps=args.max_steps,
            feedback_tier="instance",     # complaint about THIS episode, no rule stated
            memory_mode="updater",        # AssertionCollector renders "" -> empty memory
            extra_rules=(rule_name,),
            notes=f"inducibility phase A {rule_name}",
        ), collector).run()
        rows_a = read_rows(run_a)
        entry["base"] = score(rows_a, rule_name)
        induced = [a.text for a in collector.assertions if a.rule_name == rule_name]
        entry["induced_all"] = induced
        entry["complaints"] = [a.complaint for a in collector.assertions
                               if a.rule_name == rule_name][:3]

        if not induced:
            # No complaint was ever generated => the agent never violated the rule with
            # empty memory. Nothing to induce from, and nothing for memory to fix.
            entry["learned"] = None
            entry["induced_text"] = None
            entry["verdict"] = "NO-HEADROOM (never violated, no assertion induced)"
            results[rule_name] = entry
            print(f"  {rule_name:26s} base_viol={entry['base']['viol_rate']} "
                  f"-> no assertion induced")
            continue

        # The first induced line is what a learner actually holds after its first
        # complaint; later lines are re-inductions of the same rule from later episodes.
        # Screening on the first is the honest single-line memory. All are kept in
        # induced_all so text variance can be inspected offline.
        entry["induced_text"] = induced[0]

        # --- phase B: the learner's OWN line as fixed memory, same tasks ---
        run_b = f"induc{args.tag}_{model_slug}_{rule_name}_B"
        SessionDriver(SessionConfig(
            run_name=run_b, persona_path=persona_path,
            stream_task_ids=[], eval_task_ids=task_ids, checkpoints=[0],
            agent=args.agent, llm=args.llm, max_steps=args.max_steps,
            memory_mode="fixed", fixed_memory="My preference: " + induced[0],
            extra_rules=(rule_name,),
            notes=f"inducibility phase B {rule_name}",
        ), NoOpUpdater()).run()
        entry["learned"] = score(read_rows(run_b), rule_name)

        # --- oracle arm for the gap this screen is about ---
        rule = RULE_REGISTRY[rule_name]()
        run_c = f"induc{args.tag}_{model_slug}_{rule_name}_C"
        SessionDriver(SessionConfig(
            run_name=run_c, persona_path=persona_path,
            stream_task_ids=[], eval_task_ids=task_ids, checkpoints=[0],
            agent=args.agent, llm=args.llm, max_steps=args.max_steps,
            memory_mode="fixed", fixed_memory="My preference: " + rule.oracle_text,
            extra_rules=(rule_name,),
            notes=f"inducibility phase C (oracle) {rule_name}",
        ), NoOpUpdater()).run()
        entry["oracle"] = score(read_rows(run_c), rule_name)
        entry["oracle_text"] = rule.oracle_text
        results[rule_name] = entry
        bv, lv = entry["base"]["viol_rate"], entry["learned"]["viol_rate"]
        ov = entry["oracle"]["viol_rate"]
        print(f"  {rule_name:26s} base={bv} learned={lv} oracle={ov}")
        print(f"      induced: {induced[0][:100]}")

    # --- verdicts ---
    for name, e in results.items():
        if e.get("verdict"):
            continue
        bv = e["base"]["viol_rate"]
        lv = e["learned"]["viol_rate"] if e["learned"] else None
        ov = e["oracle"]["viol_rate"] if e.get("oracle") else None
        if bv is None or lv is None:
            e["verdict"] = "UNMEASURED (vacuous)"
        elif bv < 0.6:
            e["verdict"] = f"NO-HEADROOM (base viol {bv} < 0.6)"
        elif lv <= 0.1:
            e["verdict"] = "INDUCIBLE"
        elif ov is not None and ov <= 0.1:
            # oracle fixes it but the learner's own words do not: this is exactly the
            # payment_note_initials failure mode -- the complaint never carried the format
            e["verdict"] = f"ORACLE-ONLY (learned {lv}, oracle {ov})"
        else:
            e["verdict"] = f"HARD (learned {lv}, oracle {ov})"

    out = OUTPUTS_DIR / f"inducibility_{model_slug}{args.tag}.json"
    out.write_text(json.dumps({"llm": args.llm,
                               "learner_llm": args.learner_llm or args.llm,
                               "agent": args.agent, "results": results},
                              ensure_ascii=False, indent=1))

    print(f"\n=== inducibility screen ({args.llm}) ===")
    print(f"{'rule':26s}{'base':>6}{'learn':>7}{'oracle':>7}  verdict")
    for name, e in sorted(results.items(), key=lambda kv: kv[1]["verdict"]):
        f = lambda x: "  n/a" if x is None else f"{x:5.2f}"
        print(f"{name:26s}{f(e['base']['viol_rate']):>6}"
              f"{f(e['learned']['viol_rate'] if e['learned'] else None):>7}"
              f"{f(e['oracle']['viol_rate'] if e.get('oracle') else None):>7}  {e['verdict']}")
    n_ind = sum(1 for e in results.values() if e["verdict"] == "INDUCIBLE")
    print(f"\nINDUCIBLE: {n_ind}/{len(results)}")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
