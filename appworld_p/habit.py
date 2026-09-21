"""Synthetic habit observations for the habitual-card comparison.
Each observation records a sampled card and binary user feedback.
The context arm receives the raw log; the control arm computes per-card acceptance rates.
Observation count is separate from the number of AppWorld evaluation episodes."""

import json
import random
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path


DEFAULT_PI_U: dict[str, float] = {
    "American Express": 0.65,
    "Wells Fargo": 0.25,
    "Chase": 0.10,
}


FAMILY_B_OPTIONS = ("American Express", "Wells Fargo", "Chase")
FAMILY_B_EVAL_TASKS = ("37a8675_2", "37a8675_3", "530b157_2",
                       "60d0b5b_1", "60d0b5b_2", "60d0b5b_3")


PI_U_SHAPE = (0.65, 0.25, 0.10)


def pi_u_for(dominant: str, options: tuple[str, ...] = FAMILY_B_OPTIONS) -> dict[str, float]:
    """Biased-random habit with `dominant` carrying the largest mass."""
    if dominant not in options:
        raise ValueError(f"{dominant!r} not in {options}")
    rest = [o for o in options if o != dominant]
    return {dominant: PI_U_SHAPE[0],
            **{o: p for o, p in zip(rest, PI_U_SHAPE[1:])}}


NOISE_TEMPLATES = (
    "spotify | added {k} songs to playlist | user: ok",
    "spotify | liked {k} songs | user: ok",
    "phone | sent text message to contact #{k} | user: ok",
    "phone | sent text message to contact #{k} | user: sign with my name next time",
    "file_system | created {k} files | user: ok",
    "simple_note | note #{k} created | user: ok",
    "todoist | added task #{k} | user: ok",
    "gmail | sent email to contact #{k} | user: keep the subject shorter",
    "gmail | sent email to contact #{k} | user: ok",
    "amazon | placed order #{k} | user: ok",
    "venmo | liked {k} transactions on my feed | user: ok",
    "venmo | added a comment to transaction #{k} | user: ok",
    "spotify | created playlist #{k} | user: make new playlists private",
    "phone | looked up {k} contacts | user: ok",
)


@dataclass
class HabitRound:
    index: int
    wanted: str
    used: str
    accepted: bool

    def render(self, tier: str) -> str:
        """One structured log line — the legitimate channel, identical for all arms."""
        if tier == "corrective":
            return (f"interaction {self.index:02d} | venmo card payment | "
                    f"paid with {self.used} card | "
                    f"user: {'ok' if self.accepted else f'I wanted my {self.wanted} card for that'}")
        if tier == "binary":
            return (f"interaction {self.index:02d} | venmo card payment | "
                    f"paid with {self.used} card | "
                    f"user: {'ok' if self.accepted else 'was unhappy with the card choice'}")
        raise ValueError(f"unknown habit tier: {tier}")


@dataclass
class HabitStream:
    """Driver-side habit process. `advance(n)` grows the observation log; the arms read
    it only through render_log() (context class) or external_stats() (control class)."""

    pi_u: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_PI_U))
    tier: str = "binary"
    seed: int = 0
    noisy: bool = False
    noise_per_round: float = 1.0
    rounds: list[HabitRound] = field(default_factory=list)
    _noise: dict[int, list[str]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)

        self._noise_rng = random.Random(self.seed + 999_983)
        self._options = list(self.pi_u)
        self._weights = [self.pi_u[o] for o in self._options]

    def advance(self, target_rounds: int) -> None:
        """Sample micro-rounds until len(rounds) == target_rounds (no LLM cost)."""
        while len(self.rounds) < target_rounds:
            i = len(self.rounds) + 1
            wanted = self._rng.choices(self._options, self._weights)[0]
            used = self._rng.choice(self._options)
            self.rounds.append(HabitRound(i, wanted, used, used == wanted))
            if self.noisy and self.noise_per_round > 0:
                hi = int(round(2 * self.noise_per_round))
                self._noise[i] = [
                    self._noise_rng.choice(NOISE_TEMPLATES).format(
                        k=self._noise_rng.randint(1, 9))
                    for _ in range(self._noise_rng.randint(0, hi))]

    def wanted_counts(self) -> Counter:
        return Counter(r.wanted for r in self.rounds)

    def target(self) -> str:
        """f*_u: the habit itself = argmax pi_u (Q1_REDESIGN 2.3). Fixed, not sample
        dependent — the preference exists before the log does; every harness has to
        infer it from the stream. Driver-side ground truth, never a harness input."""
        return max(self.pi_u, key=lambda o: (self.pi_u[o], o))

    def empirical_mode(self) -> str | None:
        """Mode of the emitted log — diagnostic only (how identifiable is the target
        at this n?), never the checker's target."""
        counts = self.wanted_counts()
        if not counts:
            return None
        top = max(counts.values())
        return sorted(o for o, c in counts.items() if c == top)[0]

    def render_log(self) -> str:
        """Full structured log — the information upper bound of the context class."""
        if not self.rounds:
            return ""
        lines = []
        for r in self.rounds:
            for noise in self._noise.get(r.index, ()):
                lines.append(f"interaction {r.index:02d}b | {noise}")
            lines.append(r.render(self.tier))
        return "\n".join(lines)

    def acceptance_rates(self) -> dict[str, tuple[int, int]]:
        """Per-option (accepts, uses) — computable from the log by a counter, in both
        tiers. This is the credit-assignment statistic f fails to do in-context."""
        uses, accepts = Counter(), Counter()
        for r in self.rounds:
            uses[r.used] += 1
            if r.accepted:
                accepts[r.used] += 1
        return {o: (accepts[o], uses[o]) for o in self._options}

    def counter_argmax(self) -> str | None:
        """What the external counter concludes. Under uniform exploration the per-option
        acceptance rate estimates pi_u(option), so its argmax recovers the habit."""
        rates = self.acceptance_rates()
        scored = [(a / u, o) for o, (a, u) in rates.items() if u]
        if not scored:
            return None
        best = max(r for r, _ in scored)
        return sorted(o for r, o in scored if r == best)[0]

    def external_stats(self) -> str:
        """Structured statistic injected by the external-counter control harness.
        The decision maker is still f; only the statistic lives outside it."""
        if not self.rounds:
            return ""
        rates = self.acceptance_rates()
        parts = [f"{o}: {a}/{u} accepted" for o, (a, u) in
                 sorted(rates.items(), key=lambda kv: (-(kv[1][0] / kv[1][1] if kv[1][1] else 0), kv[0]))]
        return ("Payment-card acceptance statistics (maintained externally over "
                f"{len(self.rounds)} past card payments): " + ", ".join(parts) +
                f"\n=> the user's usual card is the {self.counter_argmax()} card.")

    def state(self) -> dict:
        return {"rounds": len(self.rounds), "tier": self.tier,
                "wanted_counts": dict(self.wanted_counts()),
                "target": self.target(), "empirical_mode": self.empirical_mode(),
                "counter_argmax": self.counter_argmax(),
                "counter_correct": self.counter_argmax() == self.target()}

    def log_lines(self) -> int:
        return len(self.render_log().splitlines()) if self.rounds else 0

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(
            {**self.state(), "pi_u": self.pi_u, "seed": self.seed, "noisy": self.noisy,
             "noise_per_round": self.noise_per_round, "log_lines": self.log_lines(),
             "log": self.render_log()}, indent=1))
