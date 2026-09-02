"""Reconstruct the ACE memory block at ANY stream position from a finished run's audit trail.

Why this exists: adding checkpoints to a finished Q3 run does not require re-running the training
stream. `ACEStyleUpdater.save()` already wrote every applied delta in order, and every step after
the delta -- id assignment, dedup, capacity prune -- is a pure function of the bullet list plus
`embed_text`, which is itself pure (char-trigram hashing, no model call). So the memory at n=3 is
recoverable exactly, and a new checkpoint costs only its eval episodes rather than a second
24-episode stream with fresh LLM sampling.

That distinction matters for correctness, not just cost: re-running the stream at temperature 0.7
would produce a DIFFERENT memory trajectory, so the new checkpoints would not lie on the same
curve as the existing ones. Replay keeps every checkpoint on one trajectory.

Verified by `--verify`, which replays to each n that the original run dumped and diffs against
the `memory_n*.md` the driver wrote at the time. Exact match on all three seeds is the licence to
trust the interpolated points.

Usage:
  python scripts/replay_ace_memory.py --run <session_dir_name> --verify
  python scripts/replay_ace_memory.py --run <session_dir_name> --at 2 4 8 --outdir <dir>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from appworld_p.ace_updater import Bullet, embed_text, cosine, render_bullets  # noqa: E402
from appworld_p.config import OUTPUTS_DIR  # noqa: E402


def _apply(bullets: list[Bullet], embeddings: dict, applied: list[dict], max_bullets: int):
    """Re-apply one episode's recorded deltas, then dedup and prune exactly as the updater does.

    Reads the `applied` list rather than re-deriving it: `applied` is what the updater actually
    committed after its own add-guard, so replaying it cannot diverge on a delta the original run
    rejected. Dedup and prune are recomputed rather than read back, because the audit trail records
    only HOW MANY were dropped, not which -- but both are deterministic given the list.
    """
    for d in applied:
        action, text = d.get("action"), (d.get("text") or "").strip()
        if not text:
            continue
        if action == "add":
            new_id = hashlib.md5(text.encode()).hexdigest()[:6]
            if any(b.id == new_id for b in bullets):
                continue
            bullets.append(Bullet(id=new_id, text=text))
            embeddings[new_id] = embed_text(text)
        elif action == "modify":
            target = str(d.get("id", "") or "")[:6]
            for b in bullets:
                if target and b.id.startswith(target):
                    b.text = text
                    embeddings[b.id] = embed_text(text)
                    break

    seen, deduped = [], []
    for b in bullets:
        emb = embeddings.get(b.id)
        if emb and any(cosine(emb, e) > 0.85 for e in seen):
            continue
        deduped.append(b)
        if emb:
            seen.append(emb)
    bullets = deduped

    if len(bullets) > max_bullets:
        for b in bullets[: len(bullets) - max_bullets]:
            embeddings.pop(b.id, None)
        bullets = bullets[-max_bullets:]
    return bullets, embeddings


def replay(run_dir: Path, max_bullets: int = 12) -> dict[int, str]:
    """-> {stream position n: rendered memory block as injected at that n}.

    The join is on the per-episode `updates` counter recorded in episodes.jsonl, because
    `history[i].episode` counts UPDATES (episodes that drew a complaint), not stream position:
    seed 0 took 17 updates over 24 episodes. Keying on stream index directly would silently
    shift every memory by the number of accepted episodes before it.
    """
    hist = json.loads((run_dir / "updater_final.json").read_text())["history"]
    by_update = {e["episode"]: e for e in hist}
    rows = [json.loads(x) for x in (run_dir / "episodes.jsonl").read_text().splitlines() if x.strip()]
    train = sorted((r for r in rows if r["phase"] == "train"), key=lambda r: r["session_index"])

    bullets: list[Bullet] = []
    embeddings: dict = {}
    out = {0: render_bullets(bullets)}
    done = 0
    for r in train:
        want = (r.get("updater_state") or {}).get("updates", done)
        # Apply every update that landed at or before this episode. Normally one per episode,
        # but an accepted episode adds none, so the loop -- not an index -- does the walking.
        while done < want:
            done += 1
            entry = by_update.get(done, {})
            if not entry.get("parse_failed"):
                bullets, embeddings = _apply(bullets, embeddings, entry.get("applied", []),
                                             max_bullets)
        out[r["session_index"]] = render_bullets(bullets)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="session dir name under outputs/sessions")
    ap.add_argument("--at", nargs="*", type=int, default=[],
                    help="stream positions to write memory files for")
    ap.add_argument("--outdir", default="", help="where to write memory_n*.md (default: the run dir)")
    ap.add_argument("--max-bullets", type=int, default=12)
    ap.add_argument("--verify", action="store_true",
                    help="diff the replay against the memory_n*.md the original run dumped")
    args = ap.parse_args()

    run_dir = OUTPUTS_DIR / "sessions" / args.run
    mem = replay(run_dir, args.max_bullets)

    if args.verify:
        ok = bad = 0
        for path in sorted(run_dir.glob("memory_n*.md")):
            n = int(path.stem.removeprefix("memory_n"))
            want = path.read_text().strip()
            got = mem.get(n, "").strip()
            if want == got:
                ok += 1
                print(f"  n={n:<3} MATCH ({len(got.splitlines())} lines)")
            else:
                bad += 1
                print(f"  n={n:<3} DIFFER\n    dumped: {want[:160]!r}\n    replay: {got[:160]!r}")
        print(f"\n{args.run}: {ok} match, {bad} differ")
        return 0 if not bad else 1

    outdir = Path(args.outdir) if args.outdir else run_dir
    outdir.mkdir(parents=True, exist_ok=True)
    for n in args.at:
        if n not in mem:
            print(f"  !! n={n} beyond the recorded stream ({max(mem)} episodes)")
            continue
        (outdir / f"memory_n{n}.md").write_text(mem[n])
        print(f"  wrote {outdir / f'memory_n{n}.md'}  ({len(mem[n].splitlines())} lines)")


if __name__ == "__main__":
    sys.exit(main() or 0)
