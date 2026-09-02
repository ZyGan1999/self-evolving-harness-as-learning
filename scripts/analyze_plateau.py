"""Is there a descent-then-plateau, and if not, which criterion fails and why?

Reports the three pre-registered criteria separately, and for the third distinguishes a missing
EFFECT from an interval too WIDE. That distinction is the whole reason this script exists: on p10's
payment_note_single_word the gap was +0.286 with criteria 1 and 2 passing, and the third failed only
because 7-9 applicable episodes cannot separate 0.29 from 0.00. "No plateau" and "not enough
episodes to see the plateau" call for opposite next steps.

Pools seeds by SUMMING counts, which is only legitimate when the per-seed rates agree; Cochran's Q
is reported so that assumption is visible rather than assumed. Q2 learned this the hard way -- its
three seeds had I^2 = 87% and no pooled number described any of them.

Usage:
  python scripts/analyze_plateau.py --tag _p12 --persona p12_plateau --seeds 0
  python scripts/analyze_plateau.py --tag _p12 --persona p12_plateau --seeds 0 1 2
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from appworld_p.config import OUTPUTS_DIR  # noqa: E402


def wilson(v, n, z=1.96):
    if not n:
        return float("nan"), float("nan")
    p = v / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / den
    return max(0.0, c - h), min(1.0, c + h)


def load(persona, slug, tag, seed, arm):
    d = OUTPUTS_DIR / "sessions" / f"q3{tag}_{slug}_{persona}_s{seed}_{arm}"
    path = d / "episodes.jsonl"
    if not path.exists():
        return None
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    ev = [r for r in rows if r.get("phase") == "eval"]
    if not ev:
        return None
    cell = defaultdict(lambda: [0, 0])
    tgc = defaultdict(lambda: [0, 0])
    for r in ev:
        tgc[r["n"]][0] += bool(r.get("tgc"))
        tgc[r["n"]][1] += 1
        for name, x in (r.get("rules") or {}).items():
            if x.get("applicable"):
                cell[(name, r["n"])][0] += x.get("satisfied") is False
                cell[(name, r["n"])][1] += 1
    return {"cell": cell, "tgc": tgc, "ns": sorted({r["n"] for r in ev}), "dir": d}


def merge(runs):
    """Sum counts across seeds."""
    cell = defaultdict(lambda: [0, 0])
    tgc = defaultdict(lambda: [0, 0])
    ns = set()
    for g in runs:
        for k, (v, d) in g["cell"].items():
            cell[k][0] += v
            cell[k][1] += d
        for n, (t, tn) in g["tgc"].items():
            tgc[n][0] += t
            tgc[n][1] += tn
        ns |= set(g["ns"])
    return {"cell": cell, "tgc": tgc, "ns": sorted(ns)}


def agg(cell, rules, n):
    v = sum(cell[(r, n)][0] for r in rules)
    d = sum(cell[(r, n)][1] for r in rules)
    return (v / d if d else None), v, d


def cochran_q(per_seed):
    """Q over per-seed log-odds at one checkpoint; None if any stratum is degenerate."""
    from math import log
    pts = []
    for v, d in per_seed:
        if d == 0 or v == 0 or v == d:
            return None, None      # Haldane correction would be needed; report as untestable
        p = v / d
        lo = log(p / (1 - p))
        w = v * (d - v) / d        # inverse variance of the log-odds
        pts.append((lo, w))
    if len(pts) < 2:
        return None, None
    wsum = sum(w for _, w in pts)
    mean = sum(lo * w for lo, w in pts) / wsum
    q = sum(w * (lo - mean) ** 2 for lo, w in pts)
    return q, len(pts) - 1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--persona", default="p12_plateau")
    ap.add_argument("--slug", default="claude-haiku-4-5-20251001")
    ap.add_argument("--tag", default="_p12")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    args = ap.parse_args()

    per_seed = {}
    for arm in ("none", "oracle", "selfevolve_rewrite"):
        got = [load(args.persona, args.slug, args.tag, s, arm) for s in args.seeds]
        have = [g for g in got if g]
        if not have:
            print(f"  (pending: {arm})")
            return
        if len(have) < len(args.seeds):
            print(f"  (partial: {arm} has {len(have)}/{len(args.seeds)} seeds)")
        per_seed[arm] = have

    arms = {a: merge(v) for a, v in per_seed.items()}
    rules = sorted({r for g in arms.values() for (r, _) in g["cell"]})
    ns = arms["selfevolve_rewrite"]["ns"]

    print("=" * 84)
    print(f"{args.persona}  seeds={args.seeds}  rules={rules}")
    print("=" * 84)
    print(f'{"n":>4s}{"selfevolve":>26s}{"TGC":>10s}')
    for n in ns:
        r, v, d = agg(arms["selfevolve_rewrite"]["cell"], rules, n)
        lo, hi = wilson(v, d)
        t, tn = arms["selfevolve_rewrite"]["tgc"][n]
        print(f'{n:>4d}{f"{r:.3f} [{lo:.2f},{hi:.2f}] {v}/{d}":>26s}{f"{t}/{tn}":>10s}')
    for arm in ("none", "oracle"):
        r, v, d = agg(arms[arm]["cell"], rules, 0)
        lo, hi = wilson(v, d)
        t, tn = arms[arm]["tgc"][0]
        print(f'{arm:>4s}{f"{r:.3f} [{lo:.2f},{hi:.2f}] {v}/{d}":>26s}{f"{t}/{tn}":>10s}')

    if len(args.seeds) > 1:
        print("\nPER SEED at the last checkpoint (pooling is only valid if these agree)")
        last = ns[-1]
        strata = []
        for s, g in zip(args.seeds, per_seed["selfevolve_rewrite"]):
            _, v, d = agg(g["cell"], rules, last)
            strata.append((v, d))
            print(f"  seed {s}: {v}/{d}" + (f" = {v/d:.3f}" if d else ""))
        q, dof = cochran_q(strata)
        if q is None:
            print("  Cochran Q: untestable (a stratum is 0/n or n/n)")
        else:
            from scipy.stats import chi2
            p = chi2.sf(q, dof)
            i2 = max(0.0, (q - dof) / q) if q > 0 else 0.0
            print(f"  Cochran Q={q:.2f} dof={dof} p={p:.4f}  I^2={i2:.0%}"
                  f"  {'HETEROGENEOUS -- do not pool' if p < 0.05 else 'homogeneous'}")

    print("\n" + "=" * 84)
    print("CRITERIA")
    print("=" * 84)
    rates = [agg(arms["selfevolve_rewrite"]["cell"], rules, n)[0] for n in ns]
    first, last_r, prev = rates[0], rates[-1], rates[-2] if len(rates) > 1 else None
    _, lv, ld = agg(arms["selfevolve_rewrite"]["cell"], rules, ns[-1])
    ora_r, ov, od = agg(arms["oracle"]["cell"], rules, 0)

    c1 = first - last_r > 0.10
    print(f"1 descent   {first:.3f} -> {last_r:.3f}  delta={first-last_r:+.3f}   "
          f"{'PASS' if c1 else 'FAIL'} (need >0.10)")
    c2 = prev is not None and abs(last_r - prev) < 0.05
    print(f"2 plateau   {prev:.3f} -> {last_r:.3f}  |delta|={abs(last_r-prev):.3f}   "
          f"{'PASS' if c2 else 'FAIL'} (need <0.05)")
    slo, shi = wilson(lv, ld)
    olo, ohi = wilson(ov, od)
    gap = last_r - ora_r
    disjoint = slo > ohi
    c3 = gap > 0.10 and disjoint
    print(f"3 > oracle  {last_r:.3f} [{slo:.2f},{shi:.2f}] vs oracle {ora_r:.3f} [{olo:.2f},{ohi:.2f}]")
    print(f"            gap={gap:+.3f}  disjoint={disjoint}   {'PASS' if c3 else 'FAIL'}")
    if not c3 and gap > 0.10:
        # The distinction this script exists for.
        need = None
        for cand in range(ld, 20 * ld + 1, max(1, ld // 4)):
            if wilson(round(last_r * cand), cand)[0] > wilson(0, cand)[1]:
                need = cand
                break
        print(f"            -> the EFFECT is there (gap {gap:+.3f}); the interval is too WIDE.")
        print(f"               denominator now {ld}"
              + (f", ~{need} would separate -> x{need/ld:.1f} rollouts or seeds" if need else ""))

    print("\n" + "=" * 84)
    if c1 and c2 and c3:
        print("*** DESCENT-THEN-PLATEAU CONFIRMED ***")
        print(f"    {first:.3f} -> {last_r:.3f}, flat over the last two checkpoints, "
              f"{gap:+.3f} above oracle with disjoint intervals.")
    else:
        failed = [n for n, ok in (("descent", c1), ("plateau", c2), ("above oracle", c3)) if not ok]
        print(f"not yet: {', '.join(failed)}")


if __name__ == "__main__":
    main()
