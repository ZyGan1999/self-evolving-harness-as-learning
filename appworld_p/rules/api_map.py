"""Logical actions -> AppWorld API matchers.

API names below are best guesses pending verification against downloaded api_docs
(run scripts/inspect_apis.py; test_api_map.py cross-checks automatically once data
is present). Each matcher lists candidate api names AND a (method, url-substring)
fallback so checkers degrade gracefully if a name is off.
"""

from dataclasses import dataclass

from ..episode import EpisodeRecord
from ..apicalls import ApiCall


@dataclass(frozen=True)
class ActionMatcher:
    app: str
    api_candidates: tuple[str, ...]
    method: str
    url_substrings: tuple[str, ...]  # fallback: all substrings must appear in url

    def matches(self, call: ApiCall) -> bool:
        if call.app != self.app or call.method != self.method:
            return False
        if call.api and call.api in self.api_candidates:
            return True
        return all(s in call.url for s in self.url_substrings)


# Verified against data/api_docs/standard (appworld data 0.2.0, scripts/inspect_apis.py):
#   gmail.send_email(email_addresses, subject, body, ...)          POST /gmail/emails
#   todoist.create_task(project_id, title, ..., due_date, ...)     POST /todoist/projects/{id}/tasks
#   file_system.create_file(file_path, content, ...)               POST /file_system/file
#   simple_note.create_note(title, content, ...)                   POST /simple_note/notes
#   venmo.create_transaction(receiver_email, amount, description, payment_card_id, ...)
#   spotify.like_song(song_id) / add_song_to_playlist(playlist_id, song_id)
#   amazon.place_order(payment_card_id, address_id)                POST /amazon/orders
#   supervisor.complete_task(answer, status)                       POST /supervisor/message
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
