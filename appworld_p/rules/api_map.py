"""Map logical preference actions to AppWorld APIs."""

from dataclasses import dataclass

from ..episode import EpisodeRecord
from ..apicalls import ApiCall


@dataclass(frozen=True)
class ActionMatcher:
    app: str
    api_candidates: tuple[str, ...]
    method: str
    url_substrings: tuple[str, ...]

    def matches(self, call: ApiCall) -> bool:
        if call.app != self.app or call.method != self.method:
            return False
        if call.api and call.api in self.api_candidates:
            return True
        return all(s in call.url for s in self.url_substrings)


ACTIONS: dict[str, ActionMatcher] = {
    "send_email": ActionMatcher("gmail", ("send_email", "send_email_from_draft"),
                                "post", ("email",)),
    "create_todoist_task": ActionMatcher("todoist", ("create_task",),
                                         "post", ("task",)),
    "create_file": ActionMatcher("file_system", ("create_file",),
                                 "post", ("file",)),
    "create_note": ActionMatcher("simple_note", ("create_note",),
                                 "post", ("note",)),
    "venmo_payment": ActionMatcher("venmo", ("create_transaction",),
                                   "post", ("transaction",)),
    "spotify_like_song": ActionMatcher("spotify", ("like_song",),
                                       "post", ("like",)),
    "spotify_add_to_playlist": ActionMatcher("spotify", ("add_song_to_playlist",),
                                             "post", ("playlist", "song")),
    "spotify_create_playlist": ActionMatcher("spotify", ("create_playlist",),
                                             "post", ("playlists",)),
    "amazon_place_order": ActionMatcher("amazon", ("place_order",),
                                        "post", ("order",)),
    "complete_task": ActionMatcher("supervisor", ("complete_task",),
                                   "post", ("message",)),
    "send_sms": ActionMatcher("phone", ("send_text_message",),
                              "post", ("messages", "text")),
}


def find_actions(ep: EpisodeRecord, action: str) -> list[ApiCall]:
    matcher = ACTIONS[action]
    return [c for c in ep.api_calls if matcher.matches(c)]
