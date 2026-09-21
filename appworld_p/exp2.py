"""Assertion ranking and memory rendering for Q2."""

import re

_STOPWORDS = frozenset("""the a an should be is are to of with my your our their its and or at as
by that this it user users me you always must never when unless any all in on for from""".split())

def _content_words(text: str) -> frozenset:
    return frozenset(w for w in re.findall(r"[a-z0-9']+", text.lower())
                     if w not in _STOPWORDS)

def rank_within_rule(group: list, pick: str) -> list:
    """Rank by induction order or mean token-set overlap, without consulting oracle text.
    Consensus ranking first deduplicates stripped, case-insensitive text; episode order breaks ties."""
    if pick == "earliest":
        return sorted(group, key=lambda a: a.episode_index)

    seen: set[str] = set()
    deduped = []
    for a in sorted(group, key=lambda a: a.episode_index):
        key = a.text.strip().lower()
        if key not in seen:
            seen.add(key)
            deduped.append(a)
    group = deduped
    words = [_content_words(a.text) for a in group]

    def centrality(i: int) -> float:
        others = [w for j, w in enumerate(words) if j != i]
        if not others:
            return 1.0
        return sum(len(words[i] & o) / len(words[i] | o) if (words[i] | o) else 1.0
                   for o in others) / len(others)

    scored = [(-centrality(i), group[i].episode_index, i) for i in range(len(group))]
    return [group[i] for _, _, i in sorted(scored)]

def _render_memory(lines: list[str], seed: int = 0) -> str:
    """Render the reported bullet-list memory format."""
    if not lines:
        return ""
    return "Things I know about this user's preferences:\n" + "\n".join(f"- {t}" for t in lines)
