"""Three-seed aggregate for S1's R-arm rising branch.

Pooling is NOT the default here, it is a conclusion that has to be earned. Seed 1 rose to
0.267 at L=150 while seed 2 fell back to 0.092, so a naive pool would be driven by seed 1 and
would report an effect no single seed's endpoint supports. The order is therefore:

  1. report each seed separately, per rule, with counts
  2. test homogeneity across seeds (Breslow-Day on the per-seed 2x2s)
  3. only if homogeneity is not rejected, report the Mantel-Haenszel pooled odds ratio
  4. if it IS rejected, say so and report the per-seed spread instead of a pooled number

Strata are seeds, not (task, seed): the analyzer's stratified test uses (task, seed) for an
arm contrast at fixed L, but the question here is whether the L=10 -> L=150 contrast is the
same effect in each seed, so the seed is the stratum whose homogeneity is in doubt.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from appworld_p.config import OUTPUTS_DIR  # noqa: E402

RUN = "exp2arms_v1_claude-haiku-4-5-20251001_p8_q2s1_R_L{L}_s{s}"
RULES = ["sms_signoff", "sms_greeting", "playlist_over_like", "venmo_private",
         "payment_note_single_word", "payment_note_lowercase",
         "payment_has_note", "sms_terse"]
# The PRIMARY metric scores all 8 rules, which is what the prereg declared ("for each of the 8
# rules, compare its violation rate at L=10 against its rate at max(L)=150"). All eight are
# administered -- they occupy lines in the injected block and consume the L budget -- so the
# burden of proof is on excluding one, not on reporting it. Both conclusions hold on 8: the
# descent is monotone in 3/3 seeds (p<4e-44) and the rise is positive in 3/3, with a SMALLER
# effect (+0.136/+0.018/+0.024 vs 6-rule +0.224/+0.023/+0.032), so 8 is the conservative report.
#
# --exclude reproduces the appendix variants. Each has a defect worth stating in the text, but
# each defect works against the conclusion rather than for it:
#   sms_terse         unsatisfiable on all four sms eval tasks (their instructions dictate the
#                     body; 0d8a4ee_3's dictated body is already 6 words against a 5-word cap).
#                     It raises the floor AND falls during the rising branch -- the only rule
#                     that moves against the others -- so it dilutes the rise.
#   payment_has_note  violations are a strict subset of payment_note_single_word's (note_only is
#                     0 in every cell), so the independent width is 7, not 8. This affects the
#                     degrees of freedom of a significance test, not the rate estimate.
EXCLUDED: tuple[str, ...] = ()
SEEDS = (1, 2, 3)
# Full grid now that seeds 2 and 3 have their descending branches. The pre-registered primary
# test still compares FLOOR to MAX, i.e. L=10 to L=150, which are named explicitly below rather
# than taken as the ends of this tuple, so widening the grid cannot move the test.
BRANCH = (0, 1, 2, 3, 5, 10, 20, 60, 150)
FLOOR_L, MAX_L = 10, 150


def read(L, s):
    """{(rule, task): [viol, applicable]} or None if that cell has not been written yet."""
    p = OUTPUTS_DIR / "sessions" / RUN.format(L=L, s=s) / "episodes.jsonl"
    if not p.exists():
        return None
    out = defaultdict(lambda: [0, 0])
    for line in p.read_text().splitlines():
        if not line.strip():
            continue
        ep = json.loads(line)
        if ep.get("phase") != "eval":
            continue
        for name, r in (ep.get("rules") or {}).items():
            if name in RULES and r.get("applicable"):
                out[(name, ep["task_id"])][0] += r.get("satisfied") is False
                out[(name, ep["task_id"])][1] += 1
    return out


def tot(cell, rules):
    if cell is None:
        return 0, 0
    rs = rules if isinstance(rules, (list, tuple, set)) else (rules,)
    return (sum(v[0] for k, v in cell.items() if k[0] in rs),
            sum(v[1] for k, v in cell.items() if k[0] in rs))


def wilson(v, n, z=1.96):
    if not n:
        return float("nan"), float("nan"), float("nan")
    p = v / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / den
    return p, max(0.0, c - h), min(1.0, c + h)


def mh_and_bd(strata):
    """Mantel-Haenszel pooled OR and a Breslow-Day homogeneity test.

    strata: [(a, b, c, d)] with a=viol at L=10, b=clean at L=10, c=viol at L=max, d=clean.
    Returns (or_mh, p_mh, bd_stat, p_bd, n_used). Zero-margin strata carry no information
    about an odds ratio and are dropped, which is reported by n_used.
    """
    from math import sqrt
    from scipy.stats import chi2, norm
    use = [(a, b, c, d) for a, b, c, d in strata
           if (a + b) and (c + d) and (a + c) and (b + d)]
    if not use:
        return float("nan"), float("nan"), float("nan"), float("nan"), 0
    # Reported as the odds of violating at L=max RELATIVE TO L=10, so a rise reads OR > 1.
    # The strata are built (viol@10, clean@10, viol@max, clean@max), whose natural MH ratio is
    # a*d / b*c = the odds at L=10 -- the inverse of what the claim is about -- so the two
    # products are swapped here rather than inverting at the end, which would turn an exact 0
    # into an inf and lose the "degenerate, cannot pool" signal.
    num = sum(b * c / (a + b + c + d) for a, b, c, d in use)
    den = sum(a * d / (a + b + c + d) for a, b, c, d in use)
    or_mh = num / den if den else float("inf")
    # MH chi-square with continuity correction
    s_obs = sum(a for a, b, c, d in use)
    s_exp = sum((a + b) * (a + c) / (a + b + c + d) for a, b, c, d in use)
    s_var = sum((a + b) * (c + d) * (a + c) * (b + d) /
                ((a + b + c + d) ** 2 * (a + b + c + d - 1))
                for a, b, c, d in use if (a + b + c + d) > 1)
    p_mh = 2 * norm.sf(abs(s_obs - s_exp) / sqrt(s_var)) if s_var > 0 else float("nan")
    # Breslow-Day: expected cell a in each stratum under the common OR, then chi-square
    # Breslow-Day needs a finite non-zero common OR to compute expected cells against. An OR of
    # exactly 0 or inf means some stratum has an empty margin after the drop above -- e.g. a
    # seed whose floor is 0/40, so the odds at L=10 are exactly 0 -- and then homogeneity is
    # UNTESTABLE rather than satisfied. Returning nan here is what makes the caller say so
    # instead of falling through to "pooled rise".
    bd = 0.0
    # Undo the swap for the expectation solve, which is written in terms of cell a.
    or_cell_a = (1.0 / or_mh) if or_mh not in (0.0, float("inf")) else or_mh
    for a, b, c, d in use:
        n1, n2, m1 = a + b, c + d, a + c
        N = n1 + n2
        if or_mh in (0.0, float("inf")) or not (0 < or_cell_a < float("inf")):
            return or_mh, p_mh, float("nan"), float("nan"), len(use)
        # Standard form: x^2 (OR-1) - x[OR(n1+m1) + (n2-m1)] + OR*n1*m1 = 0, in terms of the
        # OR for cell a, hence or_cell_a rather than the reported (inverted) or_mh.
        A = or_cell_a - 1
        B = -(or_cell_a * (n1 + m1) + (n2 - m1))
        C = or_cell_a * n1 * m1
        if abs(A) < 1e-12:
            x = n1 * m1 / N
        else:
            disc = max(0.0, B * B - 4 * A * C)
            x = (-B - sqrt(disc)) / (2 * A)
            lo, hi = max(0, m1 - n2), min(n1, m1)
            if not (lo - 1e-9 <= x <= hi + 1e-9):
                x = (-B + sqrt(disc)) / (2 * A)
            x = min(max(x, lo), hi)
        e = [x, n1 - x, m1 - x, n2 - m1 + x]
        if min(e) <= 0:
            continue
        v = 1.0 / sum(1.0 / q for q in e)
        bd += (a - x) ** 2 / v
    dof = max(1, len(use) - 1)
    return or_mh, p_mh, bd, chi2.sf(bd, dof), len(use)


def main() -> None:
    import argparse
    from scipy.stats import fisher_exact
    ap = argparse.ArgumentParser()
    # The primary metric scores all 8. This flag reproduces the appendix variants without
    # editing the module constant, so the two口径 cannot silently diverge between runs.
    ap.add_argument("--exclude", nargs="*", default=list(EXCLUDED), choices=RULES,
                    help="rules to drop from the aggregate (default: none = the 8-rule primary)")
    args = ap.parse_args()
    excluded = tuple(args.exclude)
    cells = {(L, s): read(L, s) for L in BRANCH for s in SEEDS}
    got = sorted({s for (L, s) in cells if cells[(L, s)] is not None})
    missing = [(L, s) for (L, s) in sorted(cells) if cells[(L, s)] is None]
    print(f"seeds with data: {got}   cells missing: {missing or 'none'}\n")

    keep = [r for r in RULES if r not in excluded]

    print("=" * 100)
    label = (f"all {len(keep)} administered rules (PRIMARY)" if not excluded
             else f"{len(keep)} rules, excluding {', '.join(excluded)} (APPENDIX VARIANT)")
    print(f"PER SEED, aggregate over {label} (rate [Wilson 95%])")
    print("=" * 100)
    print(f"{'seed':>5}  " + "".join(f"{'L='+str(L):>14}" for L in BRANCH))
    for s in got:
        row = ""
        for L in BRANCH:
            v, n = tot(cells[(L, s)], keep)
            if not n:
                row += f"{'--':>14}"
            else:
                row += f"{f'{v/n:.3f}':>14}"
        print(f"{s:>5}  {row}")

    print(f"\n{'=' * 100}")
    print("PRE-REGISTERED PRIMARY, per rule: own floor L=10 vs own L=150, each seed separately")
    print("=" * 100)
    verdicts = {}
    for r in RULES:
        line, sig = f"{r:26s}", []
        for s in got:
            av, an = tot(cells[(FLOOR_L, s)], r)
            bv, bn = tot(cells[(MAX_L, s)], r)
            if not an or not bn:
                line += f"{'--':>20}"
                continue
            p = fisher_exact([[av, an - av], [bv, bn - bv]])[1]
            up = bv / bn > av / an
            sig.append(p < 0.05 and up)
            mark = "*" if p < 0.05 and up else (" " if p >= 0.05 else "v")
            line += f"{f'{av}/{an}>{bv}/{bn} p={p:.3f}{mark}':>20}"
        verdicts[r] = sig
        print(line)
    print(f"\n{'':26s}" + "".join(f"{'seed '+str(s):>20}" for s in got))
    print("  * = significant rise    v = significant FALL    blank = n.s.")

    print(f"\n{'=' * 100}")
    print("HOMOGENEITY FIRST, then pooling. A pooled number is reported only where Breslow-Day")
    print("does not reject a common effect across seeds; otherwise the spread is the finding.")
    print("=" * 100)
    print(f"{'rule':26s} {'OR_MH':>8} {'p_MH':>9} {'BreslowDay':>11} {'p_BD':>8} {'strata':>7}  verdict")
    agg_label = f"AGGREGATE({len(keep)})"
    for r in RULES + [agg_label]:
        rules = keep if r == agg_label else r
        strata = []
        for s in got:
            av, an = tot(cells[(FLOOR_L, s)], rules)
            bv, bn = tot(cells[(MAX_L, s)], rules)
            if an and bn:
                strata.append((av, an - av, bv, bn - bv))
        if len(strata) < 2:
            print(f"{r:26s} {'need >=2 seeds':>38}")
            continue
        or_mh, p_mh, bd, p_bd, nu = mh_and_bd(strata)
        if nu == 0:
            v = "no informative stratum (all cells 0)"
        elif p_bd != p_bd:
            # nan BD: the common OR is 0 or inf because some seed's floor is exactly 0/n, so
            # homogeneity is untestable. That is not licence to pool.
            v = "homogeneity UNTESTABLE (degenerate OR) -- do not pool"
        elif p_bd < 0.05:
            v = "HETEROGENEOUS -- do not pool"
        elif p_mh == p_mh and p_mh < 0.05:
            v = f"pooled rise, OR={or_mh:.2f}"
        else:
            v = "pooled n.s."
        f = lambda x: "  nan" if x != x else f"{x:.3f}"
        print(f"{r:26s} {f(or_mh):>8} {f(p_mh):>9} {f(bd):>11} {f(p_bd):>8} {nu:>7}  {v}")

    print(f"\n{'=' * 100}")
    print("DESCENDING branch: does the fall from L=0 to the floor replicate?")
    print("=" * 100)
    from scipy.stats import fisher_exact as _fe
    for s in got:
        av, an = tot(cells[(0, s)], keep)
        bv, bn = tot(cells[(FLOOR_L, s)], keep)
        rates = [tot(cells[(L, s)], keep) for L in BRANCH if BRANCH.index(L) <= BRANCH.index(FLOOR_L)]
        seq = [v / n for v, n in rates if n]
        mono = all(seq[i] > seq[i + 1] for i in range(len(seq) - 1))
        print(f"  seed {s}: {av}/{an}={av/an:.3f} -> {bv}/{bn}={bv/bn:.3f}  "
              f"p={_fe([[av, an - av], [bv, bn - bv]])[1]:.2e}  monotone_down={mono}")

    print(f"\n{'=' * 100}")
    print("SAMPLING-NOISE BASELINE. At L=0 and L=1 the block is 0 or 1 lines, so the (seed, L)")
    print("shuffle is the identity and all three seeds run a BYTE-IDENTICAL block -- their spread")
    print("there is pure LLM sampling. At L=2 two lines admit two permutations, so seeds 1 and 3")
    print("collide. Only L>=3 gives three distinct orderings. Any seed spread at high L has to")
    print("clear this baseline before it can be attributed to line order.")
    print("=" * 100)
    print(f"{'L':>5}{'seed1':>9}{'seed2':>9}{'seed3':>9}{'mean':>9}{'range':>9}   distinct blocks")
    for L in BRANCH:
        rs = []
        for s in got:
            v, n = tot(cells[(L, s)], keep)
            rs.append(v / n if n else None)
        if any(r is None for r in rs) or len(rs) < 3:
            continue
        nb = {0: "1 (identical)", 1: "1 (identical)", 2: "2"}.get(L, "3")
        flag = "  <-- baseline" if L in (0, 1) else ""
        print(f"{L:>5}{rs[0]:>9.3f}{rs[1]:>9.3f}{rs[2]:>9.3f}"
              f"{sum(rs)/3:>9.3f}{max(rs)-min(rs):>9.3f}   {nb}{flag}")

    print(f"\n{'=' * 100}")
    print("Per-seed peak location. Reported as an OBSERVATION, not a test: choosing the peak")
    print("after seeing the data is the axis-search that amendment 1 records, and the")
    print("pre-registered comparison is against max(L)=150, not against argmax.")
    print("=" * 100)
    for r in ("sms_signoff", "sms_greeting", "playlist_over_like"):
        for s in got:
            av, an = tot(cells[(FLOOR_L, s)], r)
            # Rising branch only. Scanning all of BRANCH would return L=0, where the block is
            # empty and every rule is violated by construction -- a trivial maximum that says
            # nothing about length.
            rising = [L for L in BRANCH if L >= FLOOR_L]
            rates = [(L, *tot(cells[(L, s)], r)) for L in rising if tot(cells[(L, s)], r)[1]]
            if not rates or not an:
                continue
            pk = max(rates, key=lambda t: t[1] / t[2])
            p = fisher_exact([[av, an - av], [pk[1], pk[2] - pk[1]]])[1]
            print(f"  {r:20s} seed {s}: floor {av}/{an} -> peak at L={pk[0]:>3} "
                  f"{pk[1]}/{pk[2]} = {pk[1]/pk[2]:.2f}  p={p:.5f}")


if __name__ == "__main__":
    main()
