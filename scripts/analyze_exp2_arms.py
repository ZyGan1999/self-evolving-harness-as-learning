"""Per-rule read-out of the R/D/N sweep.

The aggregate violation rate is not the unit of analysis: two of p6_q2's five preferences
have an oracle compliance of 0.00, so they contribute a near-constant violation to every cell
and flatten whatever the other three do. The aggregate ALSO moves its own denominator,
because which rules are applicable depends on what the agent chose to do. So everything here
is reported per rule, over that rule's own applicable episodes.

What the design predicts, and what this script is built to falsify:
  R  flat or falling in L  -- redundancy, the earlier null
  N  flat in L             -- a constant noise share does not scale
  D  falling then RISING   -- relevance d/L decays, so the on-topic lines lose the block

A rate alone hides the thing worth knowing, because two different failures share it. The
`--modes` read-out splits each violation into how it failed:

  omit   the slot is empty -- the preference never fired
  value  the slot is filled and the FORM is right, but the argument is wrong

They are not interchangeable. omit says the rule was not retrieved, which is what a decaying
defended share predicts. value says it WAS retrieved and then bound to the wrong entity, which
a share cannot explain and which is how the long cells fail. Collapsing them turns two
mechanisms into one curve.
"""

import argparse
import json
import re
import sys
from math import comb, sqrt
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from appworld_p.config import OUTPUTS_DIR, TASK_POOL_PATH  # noqa: E402

SESSIONS = OUTPUTS_DIR / "sessions"


def read_cell(run_name: str) -> dict:
    """Per-rule (violations, applicable) plus TGC for one session directory."""
    path = SESSIONS / run_name / "episodes.jsonl"
    if not path.exists():
        return {}
    per: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    modes: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    # Same keying, split by task. With --rollouts K the cell holds K exchangeable draws per
    # task, and the two tasks are NOT one population (0/4 vs 3/4 was seen at R L=5), so the
    # cell rate alone averages over that split. Rollouts within a task are exchangeable and
    # may be tested; across tasks, 2 clusters cannot support a cluster-robust test, so the
    # per-task rates are reported and required to agree instead of pooled into one n.
    by_task: dict[tuple, list[int]] = defaultdict(lambda: [0, 0])
    tgc = [0, 0]
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        ep = json.loads(line)
        if ep.get("phase") != "eval":
            continue
        tgc[0] += bool(ep.get("tgc"))
        tgc[1] += 1
        for name, r in (ep.get("rules") or {}).items():
            if r.get("applicable"):
                bad = r.get("satisfied") is False
                per[name][0] += bad
                per[name][1] += 1
                by_task[(name, ep.get("task_id"))][0] += bad
                by_task[(name, ep.get("task_id"))][1] += 1
                clf = MODE_CLASSIFIERS.get(name)
                if bad and clf:
                    modes[name][clf(_artifact(r.get("detail") or ""))] += 1
    return {"per_rule": dict(per), "modes": {k: dict(v) for k, v in modes.items()},
            "by_task": dict(by_task), "tgc": tgc}


def fmt(v: int, n: int) -> str:
    return f"{v}/{n}" if n else "  -"


# Splitting omit from value needs to know what "the slot fired" looks like, which is a fact
# about each rule's artifact and cannot be read off the rate. Only rules whose checker can
# fail BOTH ways appear here: payment_note_initials tests form alone (any two bracketed
# initials pass), so a wrong-value failure is not observable through it and classifying it
# would invent a distinction the checker never drew.
_SIGNOFF_FIRED = re.compile(r"[-–—~]\s*[A-Za-z][\w.']*\s*$")

def _mode_sms_signoff(artifact: str) -> str:
    return "value" if _SIGNOFF_FIRED.search(artifact) else "omit"


MODE_CLASSIFIERS = {"sms_signoff": _mode_sms_signoff}


# Hand-rolled because this env has no scipy, and because n per cell is small enough that the
# normal approximation is wrong in the direction that flatters the hypothesis: a 0/4 cell has
# a Wald interval of zero width. Fisher and Wilson both stay honest at the boundary.
def fisher_2x2(a: int, b: int, c: int, d: int) -> float:
    """Two-sided Fisher exact p, by summing every table at most as probable as observed."""
    n, r1, r2, c1 = a + b + c + d, a + b, c + d, a + c
    if not (r1 and r2 and c1 and n - c1):
        return 1.0

    def prob(x: int) -> float:
        return comb(r1, x) * comb(r2, c1 - x) / comb(n, c1)

    obs = prob(a)
    return min(1.0, sum(prob(x) for x in range(max(0, c1 - r2), min(r1, c1) + 1)
                        if prob(x) <= obs * (1 + 1e-9)))


def _hypergeom(N: int, V: int, k: int) -> tuple[int, list[float]]:
    """(lo, pmf) for the count of violations landing in the k draws of one arm, given the
    stratum's margins: k of N draws are arm R and V of N are violations."""
    lo, hi = max(0, k - (N - V)), min(k, V)
    tot = comb(N, k)
    return lo, [comb(V, x) * comb(N - V, k - x) / tot for x in range(lo, hi + 1)]


def stratified_exact(strata: list[tuple[int, int, int, int]]) -> dict:
    """Exact conditional test of one common arm effect across independent strata, each
    (r_viol, r_n, d_viol, d_n). A stratum is a task (or task x seed): rollouts inside one
    are exchangeable draws, rollouts across tasks are not, so the arm contrast is pooled by
    conditioning on each stratum's margins rather than by concatenating the draws. T is the
    total violations in R; small T is the pre-registered direction (D > R)."""
    off, dist, t_obs = 0, [1.0], 0
    for rv, rn, dv, dn in strata:
        lo, pmf = _hypergeom(rn + dn, rv + dv, rn)
        new = [0.0] * (len(dist) + len(pmf) - 1)
        for i, a in enumerate(dist):
            if a:
                for j, b in enumerate(pmf):
                    new[i + j] += a * b
        dist, off, t_obs = new, off + lo, t_obs + rv
    exp = sum((off + i) * p for i, p in enumerate(dist))
    p_one = sum(p for i, p in enumerate(dist) if off + i <= t_obs)
    ref = dist[t_obs - off]
    p_two = sum(p for p in dist if p <= ref * (1 + 1e-9))
    return {"t": t_obs, "exp": exp, "p_one": min(1.0, p_one), "p_two": min(1.0, p_two),
            "strata": len(strata)}


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p, den = k / n, 1 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    half = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, mid - half), min(1.0, mid + half))


def ci_str(k: int, n: int) -> str:
    lo, hi = wilson(k, n)
    return f"{k}/{n}={k/n:.2f} [{lo:.2f},{hi:.2f}]" if n else f"{k}/{n}"


def _artifact(detail: str) -> str:
    """Checkers report the offending artifact as a repr after ': '. Detail carries only its
    tail, which is all the suffix rules need and is why this is a tail match."""
    m = re.search(r": '(.*)'\s*$", detail)
    return m.group(1) if m else ""


def _short(rule: str) -> str:
    """Keep the discriminating tail: payment_note_initials and payment_note_category share
    their first 13 characters, so a plain truncation renders them as the same column."""
    parts = rule.split("_")
    return parts[-1][:13] if len(parts) > 2 else rule[:13]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", default="anthropic:claude-haiku-4-5-20251001")
    ap.add_argument("--persona", default="p6_q2")
    ap.add_argument("--arms", nargs="+", default=["R", "D", "N"])
    ap.add_argument("--l-values", nargs="+", type=int,
                    default=[1, 2, 3, 5, 20, 60, 150])
    ap.add_argument("--seeds", nargs="+", type=int, default=[1])
    ap.add_argument("--tag", default="_v1")
    ap.add_argument("--out-suffix", default="")
    ap.add_argument("--modes", action="store_true",
                    help="split each violation into omit vs wrong-value")
    ap.add_argument("--stats", action="store_true",
                    help="Wilson intervals and Fisher exact on the arm and mode contrasts")
    args = ap.parse_args()
    slug = args.llm.split(":")[-1].replace(".", "_")

    cells: dict[tuple, dict] = {}
    for arm in args.arms:
        for L in args.l_values:
            for s in args.seeds:
                run = (f"exp2arms{args.tag}{args.out_suffix}_{slug}_{args.persona}"
                       f"_{arm}_L{L}_s{s}")
                c = read_cell(run)
                if c:
                    cells[(arm, L, s)] = c
    if not cells:
        print("no completed cells yet")
        return

    rules = sorted({r for c in cells.values() for r in c["per_rule"]})
    print(f"cells found: {len(cells)}  |  pooling seeds {args.seeds}\n")
    for arm in args.arms:
        print(f"===== arm {arm}")
        head = "".join(f"{_short(r):>15}" for r in rules)
        print(f"{'L':>5}{head}{'TGC':>10}{'aggr':>8}")
        for L in args.l_values:
            got = [cells[(arm, L, s)] for s in args.seeds if (arm, L, s) in cells]
            if not got:
                continue
            row, tv, ta = "", 0, 0
            for r in rules:
                v = sum(c["per_rule"].get(r, [0, 0])[0] for c in got)
                n = sum(c["per_rule"].get(r, [0, 0])[1] for c in got)
                tv, ta = tv + v, ta + n
                row += f"{(fmt(v, n) + (f' {v/n:.2f}' if n else '')):>15}"
            tg = sum(c["tgc"][0] for c in got), sum(c["tgc"][1] for c in got)
            print(f"{L:>5}{row}{fmt(*tg):>10}"
                  f"{(f'{tv/ta:.2f}' if ta else '-'):>8}")
        print()

    if args.stats:
        def pooled(arm, L, rule):
            g = [cells[(arm, L, s)] for s in args.seeds if (arm, L, s) in cells]
            v = sum(c["per_rule"].get(rule, [0, 0])[0] for c in g)
            n = sum(c["per_rule"].get(rule, [0, 0])[1] for c in g)
            return v, n

        # Cells with L < d are excluded from the R-alone and mode contrasts. d is the number of
        # rules the eval set covers, so below it the block cannot state every rule and an
        # omission is a coverage deficit rather than a length or dilution effect -- a different
        # cause sharing one label. d is a design constant read off the pool, NOT the argmin of
        # the observed curve: picking the floor after seeing the data is what inflates these p
        # values, and doing it here would move this from p=0.0076 to p=0.14 by choice of range.
        d = len(json.loads(TASK_POOL_PATH.read_text())["pools"][args.persona]["eval_rule_cover"])
        at_or_above = [L for L in args.l_values if L >= d]
        def strata_of(rule, L):
            """[(r_viol, r_n, d_viol, d_n)] with one entry per (task, seed) that has draws in
            both arms. At K=1 an entry is 1v1, the informative ones are fair coins and the test
            reduces to a sign test -- which is what the old one-draw data always needed."""
            out = []
            for s in args.seeds:
                rc, dc = cells.get(("R", L, s), {}), cells.get(("D", L, s), {})
                rb, db = rc.get("by_task", {}), dc.get("by_task", {})
                for (nm, tid), (rv, rn) in sorted(rb.items()):
                    if nm != rule:
                        continue
                    dv, dn = db.get((nm, tid), (0, 0))
                    if rn and dn:
                        out.append((rv, rn, dv, dn))
            return out

        def by_task(arm, L, rule):
            """(task_id -> (viol, n)) pooled over seeds, keeping the task split."""
            out: dict[str, list[int]] = defaultdict(lambda: [0, 0])
            for s in args.seeds:
                for (nm, tid), (v, n) in cells.get((arm, L, s), {}).get("by_task", {}).items():
                    if nm == rule:
                        out[tid][0] += v
                        out[tid][1] += n
            return {k: tuple(v) for k, v in out.items()}

        for rule in sorted(MODE_CLASSIFIERS):
            print(f"===== stats | {rule}   (Wilson 95%, Fisher exact two-sided)")
            print("  PRE-REGISTERED: D > R at equal L. Everything below it is exploratory.")
            print(f"  d={d} rules covered; L<d dropped below (coverage deficit, not dilution).")
            for L in args.l_values:
                (rv, rn), (dv, dn) = pooled("R", L, rule), pooled("D", L, rule)
                if rn and dn:
                    st = strata_of(rule, L)
                    se = stratified_exact(st) if st else None
                    # The pooled Fisher is shown for continuity with the earlier reports and is
                    # NOT the test: it pools (task x seed) draws into one 2x2. The stratified
                    # exact p is the primary, one-sided in the pre-registered direction D > R.
                    tail = (f" | strat p1={se['p_one']:.4f} p2={se['p_two']:.4f}"
                            f" T={se['t']} E={se['exp']:.1f} k={se['strata']}" if se else "")
                    print(f"    L={L:<4} R {ci_str(rv, rn):<24} D {ci_str(dv, dn):<24}"
                          f" pooled={fisher_2x2(rv, rn - rv, dv, dn - dv):.3f}{tail}")
            # Per-task, and the arm contrast tested WITHIN each task. Rollouts inside one task
            # are exchangeable draws, so a 2x2 on them is valid; the two tasks are separate
            # tests that must agree. Pooling them and reporting one n would treat correlated
            # draws as independent and overstate the result, so the pooled p above is only
            # meaningful when each cell is one draw per task.
            tasks = sorted({t for L in args.l_values for a in ("R", "D", "N")
                            for t in by_task(a, L, rule)})
            if tasks and max(len(by_task("R", L, rule)) and
                             max((n for _, n in by_task("R", L, rule).values()), default=0)
                             for L in args.l_values) > 1:
                print("  -- per task (K rollouts each); the two tasks are separate tests")
                for L in args.l_values:
                    rt, dt = by_task("R", L, rule), by_task("D", L, rule)
                    for t in tasks:
                        rv, rn = rt.get(t, (0, 0))
                        dv, dn = dt.get(t, (0, 0))
                        if not rn and not dn:
                            continue
                        p = (f"p={fisher_2x2(rv, rn - rv, dv, dn - dv):.3f}"
                             if rn and dn else "")
                        print(f"    L={L:<4} {t:<11} R {fmt(rv, rn):>7}  D {fmt(dv, dn):>7}"
                              f"  {p}")
            ends = [L for L in at_or_above if pooled("R", L, rule)[1]]
            if len(ends) > 1:
                (av, an), (bv, bn) = pooled("R", ends[0], rule), pooled("R", ends[-1], rule)
                print(f"    R alone, L={ends[0]} vs L={ends[-1]}: {ci_str(av, an)} vs "
                      f"{ci_str(bv, bn)}  p={fisher_2x2(av, an - av, bv, bn - bv):.3f}")
            tally = {a: defaultdict(int) for a in args.arms}
            for (arm, L, s), c in cells.items():
                if L < d:
                    continue
                for k, v in (c.get("modes", {}).get(rule) or {}).items():
                    tally[arm][k] += v
            print("    mode totals: " + " | ".join(
                f"{a} omit={tally[a]['omit']} value={tally[a]['value']}" for a in args.arms))
            if all(a in tally for a in ("R", "D")):
                r, d = tally["R"], tally["D"]
                print(f"    R vs D on mode:  p="
                      f"{fisher_2x2(r['omit'], r['value'], d['omit'], d['value']):.4f}")
            print()

    if args.modes:
        for rule in sorted(MODE_CLASSIFIERS):
            print(f"===== failure mode | {rule}   (omit = slot empty, "
                  f"value = form right, argument wrong)")
            print(f"{'L':>5}" + "".join(f"{a:>18}" for a in args.arms))
            for L in args.l_values:
                cs = ""
                for arm in args.arms:
                    got = [cells[(arm, L, s)] for s in args.seeds if (arm, L, s) in cells]
                    m: dict[str, int] = defaultdict(int)
                    for c in got:
                        for k, v in (c.get("modes", {}).get(rule) or {}).items():
                            m[k] += v
                    cs += f"{('.' if not got else ' '.join(f'{k}x{v}' for k, v in sorted(m.items())) or 'clean'):>18}"
                if cs.strip(" ."):
                    print(f"{L:>5}{cs}")
            print()

    print("Read the per-rule columns, not 'aggr'. A U in D means some rule's violation rate")
    print("falls from L=d then rises as its share of the block decays; R and N are the")
    print("controls that say the rise needs falling RELEVANCE, not length and not noise.")
    print("Then read --modes: a rate that rises by 'omit' and one that rises by 'value' are")
    print("different failures, and only the first is what a decaying defended share predicts.")


if __name__ == "__main__":
    main()
