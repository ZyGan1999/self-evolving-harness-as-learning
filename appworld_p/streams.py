"""Interaction-stream construction shared by experiment runners."""

import random


def build_stream(pool: dict, n_train: int, seed: int) -> list[str]:
    """Interaction stream of length n_train, sampled WITH replacement once the
    pool is exhausted (a user asks similar things repeatedly — and the eligible
    write-task pool in train/dev is small)."""
    rng = random.Random(seed)
    base = list(pool["stream_task_ids"])
    rng.shuffle(base)
    if n_train <= len(base):
        return base[:n_train]
    extra = [rng.choice(base) for _ in range(n_train - len(base))]
    return base + extra
