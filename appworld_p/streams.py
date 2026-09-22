"""Interaction-stream construction shared by experiment runners."""

import random


def build_stream(pool: dict, n_train: int, seed: int) -> list[str]:
    """Build an interaction stream, sampling with replacement after exhausting the pool."""
    rng = random.Random(seed)
    base = list(pool["stream_task_ids"])
    rng.shuffle(base)
    if n_train <= len(base):
        return base[:n_train]
    extra = [rng.choice(base) for _ in range(n_train - len(base))]
    return base + extra
