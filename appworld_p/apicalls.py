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
from contextlib import contextmanager
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
    succeeded: bool | None = None
    transaction_id: int | None = None

    def arg(self, *names: str, default: Any = None) -> Any:
        """First present argument among candidate names (API arg naming may vary)."""
        for name in names:
            if name in self.arguments and self.arguments[name] not in (None, ""):
                return self.arguments[name]
        return default


def _route_to_regex(path_template: str) -> re.Pattern:
    pattern = re.sub(r"\{(\w+)\}", r"(?P<\1>[^/]+)", path_template.rstrip("/") or "/")
    return re.compile(f"^{pattern}$")


@lru_cache(maxsize=None)
def _load_route_table(app: str) -> list[tuple[str, str, re.Pattern]]:
    """[(method, api_name, compiled_route)] for one app, from standard api_docs."""
    doc_path = APPWORLD_DATA_DIR / "api_docs" / "standard" / f"{app}.json"
    if not doc_path.exists():
        return []
    docs = json.loads(doc_path.read_text())

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

    for candidate in (url.rstrip("/") or "/", route.rstrip("/") or "/"):
        for m, api_name, regex in _load_route_table(app):
            if m != method:
                continue
            match = regex.match(candidate)
            if match:
                return ApiCall(app=app, api=api_name,
                               arguments={**match.groupdict(), **data},
                               method=method, url=url,
                               succeeded=record.get("succeeded"),
                               transaction_id=record.get("transaction_id"))
    return ApiCall(app=app, api="", arguments=data, method=method, url=url,
                   succeeded=record.get("succeeded"),
                   transaction_id=record.get("transaction_id"))


@contextmanager
def track_payment_outcomes(requester):
    """Attach actual Venmo execution outcomes to AppWorld's attempt log.

    Keep failed attempts: habitual-card scoring intentionally uses the first
    attempted card. Running-total scoring separately selects successful payments.
    """
    original = requester._request

    def request(_app_name, _api_name, *args, **kwargs):
        start = len(requester.request_tracker.requests)
        response = None
        try:
            response = original(_app_name, _api_name, *args, **kwargs)
            return response
        finally:
            if (_app_name, _api_name) == ("venmo", "create_transaction"):
                payload = {}
                if response is not None and response.status_code == 200:
                    try:
                        payload = response.json()
                    except (ValueError, TypeError):
                        pass
                transaction_id = payload.get("transaction_id") if isinstance(payload, dict) else None
                success = isinstance(transaction_id, int) and not isinstance(transaction_id, bool)
                for record in requester.request_tracker.requests[start:]:
                    record["succeeded"] = success
                    record["transaction_id"] = transaction_id if success else None

    requester._request = request
    try:
        yield
    finally:
        requester._request = original


def load_api_calls(jsonl_path: str | Path, keep_rejected: bool = False) -> list[ApiCall]:
    """Load resolved API attempts, optionally excluding schema-invalid requests."""
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
