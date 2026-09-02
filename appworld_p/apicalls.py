"""Parse AppWorld api_calls.jsonl into semantic (app, api, arguments) records.

Raw log lines look like: {"method": "post", "url": "/venmo/transactions", "data": {...}}
The URL's first path segment is the app; the rest is the API route. We resolve
routes to API names using the downloaded standard api_docs
({APPWORLD_ROOT}/data/api_docs/standard/{app}.json), which include each API's
`path` and `method`. Path params embedded in the URL are recovered from the
route template and merged into arguments.
"""

import json
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

from .config import APPWORLD_DATA_DIR


@dataclass
class ApiCall:
    app: str
    api: str
    arguments: dict[str, Any] = field(default_factory=dict)
    method: str = ""
    url: str = ""

    def arg(self, *names: str, default: Any = None) -> Any:
        """First present argument among candidate names (API arg naming may vary)."""
        for name in names:
            if name in self.arguments and self.arguments[name] not in (None, ""):
                return self.arguments[name]
        return default


def _route_to_regex(path_template: str) -> re.Pattern:
    # "/transactions/{transaction_id}/like" -> ^/transactions/(?P<transaction_id>[^/]+)/like$
    pattern = re.sub(r"\{(\w+)\}", r"(?P<\1>[^/]+)", path_template.rstrip("/") or "/")
    return re.compile(f"^{pattern}$")


@lru_cache(maxsize=None)
def _load_route_table(app: str) -> list[tuple[str, str, re.Pattern]]:
    """[(method, api_name, compiled_route)] for one app, from standard api_docs."""
    doc_path = APPWORLD_DATA_DIR / "api_docs" / "standard" / f"{app}.json"
    if not doc_path.exists():
        return []
    docs = json.loads(doc_path.read_text())
    # standard docs: {api_name: {"path": ..., "method": ..., ...}} or a list of such dicts.
    table = []
    items = docs.items() if isinstance(docs, dict) else [(d.get("api_name", d.get("name", "")), d) for d in docs]
    for api_name, doc in items:
        if not isinstance(doc, dict):
            continue
        path, method = doc.get("path"), doc.get("method")
        if not path or not method:
            continue
        table.append((method.lower(), api_name, _route_to_regex(path)))
    return table


def was_rejected(call: "ApiCall") -> bool:
    """True if the API cannot have accepted this call (a required parameter is absent).

    Must be applied on BOTH read paths: the live driver builds ApiCalls from the
    in-memory request tracker, so filtering only inside load_api_calls would leave
    every real run unfiltered.
    """
    if not call.api:
        return False
    return bool(_required_params(call.app, call.api) - set(call.arguments))


@lru_cache(maxsize=None)
def _required_params(app: str, api: str) -> frozenset[str]:
    """Names the API rejects the call without ('Validation error: field required')."""
    doc_path = APPWORLD_DATA_DIR / "api_docs" / "standard" / f"{app}.json"
    if not doc_path.exists():
        return frozenset()
    docs = json.loads(doc_path.read_text())
    items = (docs.items() if isinstance(docs, dict)
             else [(d.get("api_name", d.get("name", "")), d) for d in docs])
    for api_name, doc in items:
        if api_name == api and isinstance(doc, dict):
            return frozenset(p["name"] for p in (doc.get("parameters") or [])
                             if isinstance(p, dict) and p.get("required")
                             and p.get("name"))
    return frozenset()


def resolve_call(record: dict[str, Any]) -> ApiCall:
    """Resolve one raw log record. Unresolvable routes keep api='' (still inspectable)."""
    method = record.get("method", "").lower()
    url = record.get("url", "")
    data = dict(record.get("data") or {})
    segments = url.lstrip("/").split("/", 1)
    app = segments[0] if segments else ""
    route = "/" + (segments[1] if len(segments) > 1 else "")
    # api_docs paths INCLUDE the app prefix ('/phone/messages/text/{phone_number}'), so
    # matching only the stripped remainder never resolved anything: every call carried
    # api='' and the checkers silently ran on the url-substring fallback alone. Try the
    # full url first, then the stripped route for any doc that omits the prefix.
    for candidate in (url.rstrip("/") or "/", route.rstrip("/") or "/"):
        for m, api_name, regex in _load_route_table(app):
            if m != method:
                continue
            match = regex.match(candidate)
            if match:
                return ApiCall(app=app, api=api_name,
                               arguments={**match.groupdict(), **data},
                               method=method, url=url)
    return ApiCall(app=app, api="", arguments=data, method=method, url=url)


def load_api_calls(jsonl_path: str | Path, keep_rejected: bool = False) -> list[ApiCall]:
    """Resolved calls, with API-rejected attempts dropped by default.

    api_calls.jsonl logs every ATTEMPT, including ones the API refused. A constrained
    function-call agent guesses parameter names, so rejects are common: in calib_famA3
    the agent first tried send_text_message(message_body=...), got 'Validation error.
    Reason: message: field required', then retried with message=... . Scoring the
    rejected attempt invented a violation ("sms lacks checksum tag: ''") for a message
    that was in fact sent with a tag -- and it did so for 2 of 6 episodes. A call missing
    a required parameter cannot have taken effect, so it must not be scored.
    """
    path = Path(jsonl_path)
    if not path.exists():
        return []
    calls = []
    for line in path.read_text().strip().splitlines():
        if not line.strip():
            continue
        call = resolve_call(json.loads(line))
        if not keep_rejected and was_rejected(call):
            continue
        calls.append(call)
    return calls
