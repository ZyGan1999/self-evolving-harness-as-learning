"""Relevant-memory construction for the reported Q2 R arm."""

import random

from .exp2 import _render_memory, rank_within_rule

def _by_rule(assertions: list, pick: str) -> dict[str, list]:
    out: dict[str, list] = {}
    for a in assertions:
        out.setdefault(a.rule_name, []).append(a)
    return {r: rank_within_rule(g, pick) for r, g in out.items()}

def _cycle(groups: dict[str, list], k: int) -> list:
    """Round-robin the first k lines across rules: every rule contributes its top line
    before any contributes a second, so coverage is complete as early as possible."""
    ranked = sorted(groups, key=lambda r: (-len(groups[r]), r))
    out, depth = [], 0
    while len(out) < k and depth < max((len(g) for g in groups.values()), default=0):
        for r in ranked:
            if depth < len(groups[r]) and len(out) < k:
                out.append(groups[r][depth])
        depth += 1
    return out

def _recycle(lines: list[str], k: int) -> list[str]:
    """Repeat a finite assertion list to fill the requested length."""
    if not lines:
        return []
    return [lines[i % len(lines)] for i in range(k)]

def build_arm_memory(arm: str, scored: list, d: int, L: int,
                     pick: str = "consensus", seed: int = 0,
                     shuffle: bool = True, cap_distinct: int | None = 55) -> tuple[str, dict]:
    """Construct the reported R arm without a donor pool."""
    if arm != "R":
        raise ValueError(f"Unknown reported Q2 arm: {arm!r}")
    if L < 0:
        raise ValueError("Memory length must be nonnegative")
    sg = _by_rule(scored, pick)
    cover = _cycle(sg, min(d, L))
    rest = max(0, L - len(cover))
    more = [a.text for a in _cycle(sg, len(cover) + rest)][len(cover):]
    if cap_distinct:
        seen = []
        for text in more:
            if text not in seen and len(seen) < cap_distinct:
                seen.append(text)
        more = seen
    on_text = [a.text for a in cover] + (_recycle(more, rest) if more else [])
    src = {a.text: a.rule_name for a in scored}
    defend = {}
    for line in on_text:
        rule = src.get(line)
        if rule:
            defend[rule] = defend.get(rule, 0) + 1
    idx = list(range(len(on_text)))
    if shuffle:
        random.Random((seed, L).__repr__()).shuffle(idx)
    lines = [on_text[i] for i in idx]
    positions = list(range(len(lines)))
    return _render_memory(lines, seed=seed), {
        "arm": arm, "L": L, "lines": len(lines),
        "on_topic": len(on_text), "off_topic": 0, "noise": 0,
        "relevance": round(len(on_text) / max(len(lines), 1), 4),
        "distinct_on": len(set(on_text)), "distinct_off": 0,
        "rules_covered": len({a.rule_name for a in cover}),
        "defenders": dict(sorted(defend.items())),
        "defended_share": {r: round(c / max(len(lines), 1), 4)
                           for r, c in sorted(defend.items())},
        "mean_on_pos": round(sum(positions) / len(positions), 2) if positions else None,
    }
