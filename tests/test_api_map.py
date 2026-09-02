"""Cross-check api_map.py guesses against downloaded api_docs (skips if data absent)."""

import json

import pytest

from appworld_p.config import APPWORLD_DATA_DIR
from appworld_p.rules.api_map import ACTIONS

DOCS_DIR = APPWORLD_DATA_DIR / "api_docs" / "standard"

pytestmark = pytest.mark.skipif(not DOCS_DIR.exists(),
                                reason="appworld data not downloaded yet")


def _api_names(app: str) -> set[str]:
    path = DOCS_DIR / f"{app}.json"
    if not path.exists():
        return set()
    docs = json.loads(path.read_text())
    if isinstance(docs, dict):
        return set(docs.keys())
    return {d.get("api_name", d.get("name", "")) for d in docs if isinstance(d, dict)}


@pytest.mark.parametrize("action", sorted(ACTIONS))
def test_action_has_real_api(action):
    matcher = ACTIONS[action]
    names = _api_names(matcher.app)
    assert names, f"no api docs for app {matcher.app}"
    matched = [c for c in matcher.api_candidates if c in names]
    assert matched, (f"{action}: none of {matcher.api_candidates} exist in "
                     f"{matcher.app} api docs; actual apis: {sorted(names)[:30]}")
