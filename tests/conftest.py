import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest  # noqa: E402

from appworld_p.apicalls import ApiCall  # noqa: E402
from appworld_p.episode import EpisodeRecord  # noqa: E402
from appworld_p.history import SessionHistory  # noqa: E402


SUPERVISOR = {"first_name": "Lena", "last_name": "Ortiz",
              "email": "lena@example.com", "phone_number": "555-0101"}


def make_episode(calls: list[ApiCall], **kwargs) -> EpisodeRecord:
    defaults = dict(task_id="t_test", instruction="test", supervisor=SUPERVISOR,
                    api_calls=calls, task_completed=True)
    defaults.update(kwargs)
    return EpisodeRecord(**defaults)


def call(app: str, api: str, method: str = "post", url: str | None = None,
         **arguments) -> ApiCall:
    return ApiCall(app=app, api=api, arguments=arguments, method=method,
                   url=url if url is not None else f"/{app}/{api}")


@pytest.fixture
def history() -> SessionHistory:
    return SessionHistory()
