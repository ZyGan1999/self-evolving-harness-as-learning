"""Which HTTP statuses the LLM client retries.

An explicit allow-list of {429,500,502,503,504,529} let a relay 520 through and killed a
32-episode collection 13 episodes in. 520-527 are Cloudflare's origin-error family and are as
transient as a 503, so the rule is the class of server-side statuses rather than a list of the
ones seen so far.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from appworld_p.llm import is_retryable  # noqa: E402


def test_retries_rate_limit_and_every_5xx() -> None:
    assert is_retryable(429)
    for status in (500, 502, 503, 504, 529, 520, 521, 522, 523, 524, 525, 526, 527, 599):
        assert is_retryable(status), status


def test_does_not_retry_client_errors_or_success() -> None:
    for status in (200, 201, 400, 401, 403, 404, 422, 499):
        assert not is_retryable(status), status
