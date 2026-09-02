"""Dump API names/signatures for the apps our checkers target, to verify api_map.py.

Usage: conda run -n appworld-p python scripts/inspect_apis.py [app ...]
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from appworld_p.config import APPWORLD_DATA_DIR  # noqa: E402

TARGET_APPS = ["gmail", "todoist", "file_system", "simple_note", "venmo", "spotify",
               "amazon", "supervisor", "splitwise", "phone"]


def main() -> None:
    apps = sys.argv[1:] or TARGET_APPS
    docs_dir = APPWORLD_DATA_DIR / "api_docs" / "standard"
    if not docs_dir.exists():
        sys.exit(f"api docs not found at {docs_dir} — run `appworld download data` first")
    for app in apps:
        path = docs_dir / f"{app}.json"
        if not path.exists():
            print(f"== {app}: NO DOC FILE ==")
            continue
        docs = json.loads(path.read_text())
        items = docs.items() if isinstance(docs, dict) else [
            (d.get("api_name", d.get("name", "?")), d) for d in docs]
        print(f"== {app} ({len(list(items))} apis) ==")
        items = docs.items() if isinstance(docs, dict) else [
            (d.get("api_name", d.get("name", "?")), d) for d in docs]
        for name, doc in items:
            if not isinstance(doc, dict):
                continue
            params = doc.get("parameters", [])
            if isinstance(params, list):
                param_names = [p.get("name", "?") if isinstance(p, dict) else str(p)
                               for p in params]
            else:
                param_names = list(params)
            print(f"  {doc.get('method', '?').upper():6s} {doc.get('path', '?'):45s} "
                  f"{name}({', '.join(param_names)})")
        print()


if __name__ == "__main__":
    main()
