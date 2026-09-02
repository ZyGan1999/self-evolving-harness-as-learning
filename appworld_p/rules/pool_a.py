"""Pool A: in-support candidate rules (style/choice preferences, pretrain-dense concepts).

Every checker is field-level programmatic (regex/param checks) — no fuzzy judgments.
"""

from ..episode import EpisodeRecord
from ..history import SessionHistory
from .api_map import find_actions
from .base import (DATE_PREFIX_RE, GREETING_RE, INITIALS_SUFFIX_RE, KEBAB_RE, Rule,
                   register, word_count)


@register
class EmailGreetingSignoff(Rule):
    name = "email_greeting_signoff"
    pool = "A"
    apps = ("gmail",)
    trigger_actions = ("send_email",)
    field_claims = ("gmail.body",)
    oracle_text = ("Every email you send for me must start with a greeting line "
                   "(e.g. 'Hi <name>,') and end with a sign-off followed by my first name.")
    correction_template = ("When you email people for me, always open with a greeting "
                           "and sign off with my first name.")

    def applicable(self, ep: EpisodeRecord) -> bool:
        return bool(find_actions(ep, "send_email"))

    def satisfied(self, ep, history):
        for call in find_actions(ep, "send_email"):
            body = str(call.arg("body", "content", "text", default=""))
            first_name = str(ep.supervisor.get("first_name", "")).strip()
            if not GREETING_RE.match(body):
                return False, f"email body lacks greeting: {body[:60]!r}"
            if first_name and first_name.lower() not in body.strip().splitlines()[-1].lower():
                return False, "email body lacks first-name sign-off"
        return True, ""


@register
class EmailSubjectShort(Rule):
    name = "email_subject_short"
    pool = "A"
    apps = ("gmail",)
    trigger_actions = ("send_email",)
    field_claims = ("gmail.subject",)
    negation_text = 'Email subjects should be long and descriptive — at least 12 words.'
    max_words = 8
    oracle_text = "Email subjects must be at most 8 words."
    correction_template = "Keep email subjects short — 8 words max."

    def applicable(self, ep):
        return bool(find_actions(ep, "send_email"))

    def satisfied(self, ep, history):
        for call in find_actions(ep, "send_email"):
            subject = str(call.arg("subject", "title", default=""))
            if word_count(subject) > self.max_words:
                return False, f"subject too long ({word_count(subject)} words): {subject!r}"
        return True, ""


@register
class TodoistDueDate(Rule):
    name = "todoist_due_date"
    pool = "A"
    apps = ("todoist",)
    trigger_actions = ("create_todoist_task",)
    field_claims = ("todoist.due_date",)
    negation_text = 'Todoist tasks you create for me must have NO due date set.'
    oracle_text = "Every Todoist task you create for me must have a due date set."
    correction_template = "Always set a due date when you create tasks in Todoist."

    def applicable(self, ep):
        return bool(find_actions(ep, "create_todoist_task"))

    def satisfied(self, ep, history):
        for call in find_actions(ep, "create_todoist_task"):
            if call.arg("due_date", "due_at", "date", "deadline") is None:
                return False, f"task created without due date: {call.arguments}"
        return True, ""


@register
class TodoistTitleCapitalized(Rule):
    name = "todoist_title_capitalized"
    pool = "A"
    apps = ("todoist",)
    trigger_actions = ("create_todoist_task",)
    field_claims = ("todoist.title",)
    negation_text = 'Todoist task titles must start with a lowercase letter.'
    oracle_text = "Todoist task titles must start with a capital letter."
    correction_template = "Capitalize the first letter of Todoist task titles."

    def applicable(self, ep):
        return bool(find_actions(ep, "create_todoist_task"))

    def satisfied(self, ep, history):
        for call in find_actions(ep, "create_todoist_task"):
            title = str(call.arg("title", "name", "content", default=""))
            if title and not title[0].isupper():
                return False, f"title not capitalized: {title!r}"
        return True, ""


@register
class FileKebabCase(Rule):
    name = "file_kebab_case"
    pool = "A"
    apps = ("file_system",)
    trigger_actions = ("create_file",)
    field_claims = ("file_system.name",)
    negation_text = 'File names you create must be in CamelCase (each word capitalized, no hyphens).'
    oracle_text = ("File names you create must be kebab-case: lowercase words "
                   "separated by hyphens (extension excluded).")
    correction_template = "Name files in kebab-case (like-this.txt)."

    def applicable(self, ep):
        return bool(find_actions(ep, "create_file"))

    def satisfied(self, ep, history):
        for call in find_actions(ep, "create_file"):
            name = str(call.arg("file_path", "name", "file_name", "path", default=""))
            stem = name.rsplit("/", 1)[-1].rsplit(".", 1)[0]
            if stem and not KEBAB_RE.match(stem):
                return False, f"file name not kebab-case: {name!r}"
        return True, ""


@register
class NoteTitleDated(Rule):
    name = "note_title_dated"
    pool = "A"
    apps = ("simple_note",)
    trigger_actions = ("create_note",)
    field_claims = ("simple_note.title",)
    negation_text = 'Note titles must NOT contain any date — keep them purely descriptive.'
    oracle_text = "Notes you create must have titles prefixed with the date as 'YYYY-MM-DD: '."
    correction_template = "Prefix note titles with the date, like '2026-07-07: ...'."

    def applicable(self, ep):
        return bool(find_actions(ep, "create_note"))

    def satisfied(self, ep, history):
        for call in find_actions(ep, "create_note"):
            title = str(call.arg("title", "name", default=""))
            if not DATE_PREFIX_RE.match(title):
                return False, f"note title not date-prefixed: {title!r}"
        return True, ""


@register
class PaymentHasNote(Rule):
    name = "payment_has_note"
    pool = "A"
    apps = ("venmo",)
    trigger_actions = ("venmo_payment",)
    field_claims = ("venmo.note",)
    negation_text = 'Venmo payments you make for me must have an EMPTY description — never write a note.'
    oracle_text = "Every Venmo payment must include a non-empty description note."
    correction_template = "Always write a description when you send money on Venmo."

    def applicable(self, ep):
        return bool(find_actions(ep, "venmo_payment"))

    def satisfied(self, ep, history):
        for call in find_actions(ep, "venmo_payment"):
            note = call.arg("description", "note", "memo")
            if not note or not str(note).strip():
                return False, "venmo payment without description"
        return True, ""


@register
class SmsSignoff(Rule):
    name = "sms_signoff"
    pool = "A"
    apps = ("phone",)
    trigger_actions = ("send_sms",)
    field_claims = ("phone.message_suffix",)
    negation_text = 'Text messages you send for me must NOT have any name sign-off at the end — just the content.'
    oracle_text = "Text messages you send for me must end with my first name as a sign-off."
    correction_template = "Sign text messages with my first name at the end."

    def applicable(self, ep):
        return bool(find_actions(ep, "send_sms"))

    def satisfied(self, ep, history):
        first_name = str(ep.supervisor.get("first_name", "")).strip()
        for call in find_actions(ep, "send_sms"):
            message = str(call.arg("message", "text", default="")).strip()
            if first_name and not message.lower().endswith(first_name.lower()):
                return False, f"sms not signed with first name: {message[-40:]!r}"
        return True, ""


@register
class VenmoPrivate(Rule):
    name = "venmo_private"
    pool = "A"
    apps = ("venmo",)
    trigger_actions = ("venmo_payment",)
    field_claims = ("venmo.private",)
    negation_text = 'Venmo transactions you make for me must always be public (never mark them private).'
    oracle_text = "Venmo transactions you make for me must always be marked private."
    correction_template = "Always mark my Venmo transactions as private."

    def applicable(self, ep):
        return bool(find_actions(ep, "venmo_payment"))

    def satisfied(self, ep, history):
        for call in find_actions(ep, "venmo_payment"):
            if not call.arg("private", default=False):
                return False, "venmo transaction not marked private"
        return True, ""


@register
class NoteTagsRequired(Rule):
    name = "note_tags_required"
    pool = "A"
    apps = ("simple_note",)
    trigger_actions = ("create_note",)
    field_claims = ("simple_note.tags",)
    negation_text = 'Notes you create for me must have NO tags at all.'
    oracle_text = "Notes you create for me must always include at least one tag."
    correction_template = "Add at least one tag when you create notes."

    def applicable(self, ep):
        return bool(find_actions(ep, "create_note"))

    def satisfied(self, ep, history):
        for call in find_actions(ep, "create_note"):
            tags = call.arg("tags")
            if not tags:
                return False, "note created without tags"
        return True, ""


@register
class PlaylistPrivate(Rule):
    name = "playlist_private"
    pool = "A"
    apps = ("spotify",)
    trigger_actions = ("spotify_create_playlist",)
    field_claims = ("spotify.playlist_visibility",)
    negation_text = 'Playlists you create for me must always be public.'
    oracle_text = "Playlists you create for me must always be private (not public)."
    correction_template = "Keep playlists you create for me private."

    def applicable(self, ep):
        return bool(find_actions(ep, "spotify_create_playlist"))

    def satisfied(self, ep, history):
        for call in find_actions(ep, "spotify_create_playlist"):
            if call.arg("is_public", default=False):
                return False, f"playlist created public: {call.arguments.get('title')!r}"
        return True, ""


@register
class PlaylistOverLike(Rule):
    name = "playlist_over_like"
    pool = "A"
    apps = ("spotify",)
    trigger_actions = ("spotify_like_song", "spotify_add_to_playlist")
    field_claims = ("spotify.save_action",)
    negation_text = "When saving music for me on Spotify, 'like' the songs instead of adding them to playlists."
    oracle_text = ("When saving music for me on Spotify, add songs to a playlist "
                   "instead of liking them.")
    correction_template = "Don't 'like' songs for me — add them to a playlist instead."

    def applicable(self, ep):
        return bool(find_actions(ep, "spotify_like_song")) or bool(
            find_actions(ep, "spotify_add_to_playlist"))

    def satisfied(self, ep, history):
        likes = find_actions(ep, "spotify_like_song")
        if likes:
            return False, f"{len(likes)} like_song call(s) used instead of playlist add"
        return True, ""


# ---------------------------------------------------------------- Q2 attribute slots
#
# Q2 needs several binary in-support preferences at once, and the five it could assemble from
# the rules above turned out to carry only two with any room: payment_has_note (0.17 baseline
# violation) and playlist_private (0.09) are already satisfied by default, so writing them
# correctly cannot help and only a REVERSED assertion moves them. That flattens the descending
# branch and leaves d=2 effectively.
#
# The constraint on new slots is that AppWorld task instructions dictate most user-visible
# text: the playlist title is given as "Spotify Recommended Songs", the file path as
# "~/backups/spotify.csv". A preference on a dictated field contradicts the instruction and
# becomes unmeasurable -- that is exactly why file_kebab_case calibrates at oracle 0.00. So the
# safe slots are the ones the task never mentions: optional boolean flags, and purely additive
# decoration (a prefix or suffix around text the task does dictate).

@register
class FileOverwriteAlways(Rule):
    """Optional boolean the task never mentions -- the cleanest kind of Q2 attribute slot."""
    name = "file_overwrite_always"
    pool = "A"
    apps = ("file_system",)
    trigger_actions = ("create_file",)
    field_claims = ("file_system.overwrite",)
    oracle_text = ("When you create a file for me, always pass overwrite=True so a stale "
                   "version is never left behind.")
    negation_text = ("When you create a file for me, never pass overwrite=True — leave any "
                     "existing file untouched.")
    correction_template = "Create files with overwrite=True so old versions are replaced."

    def applicable(self, ep):
        return bool(find_actions(ep, "create_file"))

    def satisfied(self, ep, history):
        for call in find_actions(ep, "create_file"):
            flag = call.arg("overwrite", default=None)
            if flag is not True and str(flag).lower() != "true":
                return False, f"create_file without overwrite=True (got {flag!r})"
        return True, ""


@register
class SmsGreeting(Rule):
    """Additive PREFIX on the message. The suffix slot is sms_signoff's; the task dictates the
    body text, so only decoration around it is measurable."""
    name = "sms_greeting"
    pool = "A"
    apps = ("phone",)
    trigger_actions = ("send_sms",)
    field_claims = ("phone.message_prefix",)
    oracle_text = ("Text messages you send for me must open with 'Hi ' followed by the "
                   "recipient's first name and a comma, before the message body.")
    negation_text = ("Text messages you send for me must start straight with the message "
                     "body — never open with a greeting.")
    correction_template = "Open text messages with a greeting like 'Hi Sam,' before the body."

    def applicable(self, ep):
        return bool(find_actions(ep, "send_sms"))

    def satisfied(self, ep, history):
        for call in find_actions(ep, "send_sms"):
            message = str(call.arg("message", "text", default="")).strip()
            # The recipient's name is not in the api call (only the number), so the checker
            # verifies the FORM of the greeting, not who it names. Requiring the right name
            # would make this a lookup task and stop measuring the preference.
            if not GREETING_RE.match(message):
                return False, f"sms lacks an opening greeting: {message[:40]!r}"
        return True, ""


@register
class PaymentNoteInitials(Rule):
    """Additive SUFFIX on the Venmo note: the task sometimes dictates the note text, so the
    preference has to sit around it rather than replace it."""
    name = "payment_note_initials"
    pool = "A"
    apps = ("venmo",)
    trigger_actions = ("venmo_payment",)
    field_claims = ("venmo.note_suffix",)
    oracle_text = ("Every Venmo payment description must end with my initials in round "
                   "brackets, for example '(J.D.)'.")
    negation_text = ("Venmo payment descriptions must never end with initials in brackets — "
                     "leave the description as it is.")
    correction_template = "End payment descriptions with my initials in brackets, like '(J.D.)'."

    def applicable(self, ep):
        return bool(find_actions(ep, "venmo_payment"))

    def satisfied(self, ep, history):
        for call in find_actions(ep, "venmo_payment"):
            note = str(call.arg("description", "note", "memo", default="")).strip()
            # Form only, same reason as sms_greeting: any two initials in brackets at the end.
            if not INITIALS_SUFFIX_RE.search(note):
                return False, f"note lacks initials suffix: {note[-30:]!r}"
        return True, ""


@register
class FileContentHeader(Rule):
    """Additive PREFIX on file content. The task dictates what goes IN the file, so the
    preference asks for a header line above it rather than changing the data."""
    name = "file_content_header"
    pool = "A"
    apps = ("file_system",)
    trigger_actions = ("create_file",)
    field_claims = ("file_system.content_prefix",)
    oracle_text = ("Files you create for me must begin with a comment line of the form "
                   "'# generated by assistant' before any other content.")
    negation_text = ("Files you create for me must start directly with the data — never add "
                     "a header comment line.")
    correction_template = ("Start created files with the header line '# generated by "
                           "assistant'.")

    def applicable(self, ep):
        # Only when content was actually written: a create_file that passes no content has no
        # first line to constrain, so judging it either way would be a different claim.
        return any(str(c.arg("content", "body", "text", default="")).strip()
                   for c in find_actions(ep, "create_file"))

    def satisfied(self, ep, history):
        for call in find_actions(ep, "create_file"):
            content = str(call.arg("content", "body", "text", default=""))
            if not content.strip():
                continue
            first = content.lstrip().splitlines()[0].strip().lower()
            if not first.startswith("#"):
                return False, f"file content lacks a header comment: {first[:40]!r}"
        return True, ""


@register
class PaymentNoteCategoryPrefix(Rule):
    """Additive PREFIX on the Venmo note, independent of the initials suffix on the same field.

    Two additive constraints can share one text field as long as they occupy opposite ends, so
    note_prefix and note_suffix are separate claims. Q2 needs the attribute count, and the
    registry's supply of untouched fields on triggerable actions is exhausted.
    """
    name = "payment_note_category"
    pool = "A"
    apps = ("venmo",)
    trigger_actions = ("venmo_payment",)
    field_claims = ("venmo.note_prefix",)
    oracle_text = ("Every Venmo payment description must start with the tag '[personal]' "
                   "before anything else.")
    negation_text = ("Venmo payment descriptions must never start with a bracketed tag — "
                     "begin with the description itself.")
    correction_template = "Start payment descriptions with the tag '[personal]'."

    def applicable(self, ep):
        return bool(find_actions(ep, "venmo_payment"))

    def satisfied(self, ep, history):
        for call in find_actions(ep, "venmo_payment"):
            note = str(call.arg("description", "note", "memo", default="")).strip()
            if not note.lower().startswith("[personal]"):
                return False, f"note lacks the category tag: {note[:30]!r}"
        return True, ""


# ------------------------------------------------- interfering distractors (Q2 rising branch)
#
# Distractor attributes exist to be written when L outgrows the evidence. For that to COST
# anything they have to constrain a field the scored rules also constrain -- a wrong assertion
# has no other channel through which to hurt. The first Q2 run excluded the one interfering
# candidate on the grounds that a self-contradictory memory would confound thin evidence with
# instruction conflict, and the result was that every distractor was about gmail, todoist,
# spotify or file_system while every scored rule was on phone.message or venmo, so writing them
# in either direction changed nothing: violation rate was flat from L=8 to L=15 (0.385, 0.438,
# 0.385) because only inert padding was being added.
#
# Instruction conflict IS how a wrong assertion hurts, so these are deliberately on the scored
# fields. They are never scored themselves, only written, and which direction the posterior
# picks decides whether a given one interferes -- which is the mechanism, not a flaw. Several
# of them rather than one keeps the rising branch from resting on a single coin flip.

@register
class PaymentNoteSingleWord(Rule):
    """Distractor: conflicts with the prefix and suffix the persona wants on the same note."""
    name = "payment_note_single_word"
    pool = "A"
    apps = ("venmo",)
    trigger_actions = ("venmo_payment",)
    field_claims = ("venmo.note_length",)
    oracle_text = "Venmo payment descriptions must be exactly one word, with no punctuation."
    negation_text = ("Venmo payment descriptions may be as long as they need to be, "
                     "punctuation included.")
    correction_template = "Keep payment descriptions to a single word."

    def applicable(self, ep):
        return bool(find_actions(ep, "venmo_payment"))

    def satisfied(self, ep, history):
        for call in find_actions(ep, "venmo_payment"):
            note = str(call.arg("description", "note", "memo", default="")).strip()
            if word_count(note) != 1:
                return False, f"note is not a single word: {note[:40]!r}"
        return True, ""


@register
class PaymentNoteNoBrackets(Rule):
    """Distractor: its positive direction forbids exactly the two decorations the persona asks
    for ('[personal]' and '(J.D.)')."""
    name = "payment_note_no_brackets"
    pool = "A"
    apps = ("venmo",)
    trigger_actions = ("venmo_payment",)
    field_claims = ("venmo.note_punctuation",)
    oracle_text = ("Venmo payment descriptions must never contain brackets or parentheses of "
                   "any kind.")
    negation_text = "Venmo payment descriptions may freely use brackets and parentheses."
    correction_template = "Leave brackets and parentheses out of payment descriptions."

    def applicable(self, ep):
        return bool(find_actions(ep, "venmo_payment"))

    def satisfied(self, ep, history):
        for call in find_actions(ep, "venmo_payment"):
            note = str(call.arg("description", "note", "memo", default=""))
            if any(ch in note for ch in "[]()"):
                return False, f"note contains brackets: {note[:40]!r}"
        return True, ""


@register
class SmsTerse(Rule):
    """Distractor: a five-word cap leaves no room for the persona's greeting plus sign-off."""
    name = "sms_terse"
    pool = "A"
    apps = ("phone",)
    trigger_actions = ("send_sms",)
    field_claims = ("phone.message_length",)
    oracle_text = "Text messages you send for me must be at most five words long."
    negation_text = "Text messages you send for me may be as long as they need to be."
    correction_template = "Keep text messages to five words or fewer."

    def applicable(self, ep):
        return bool(find_actions(ep, "send_sms"))

    def satisfied(self, ep, history):
        for call in find_actions(ep, "send_sms"):
            message = str(call.arg("message", "text", default=""))
            if word_count(message) > 5:
                return False, f"message longer than five words: {word_count(message)}"
        return True, ""


# Four more, because at L=15 only 2 of the first 4 landed in their harmful direction and the
# rising branch came out at +0.125 with a one-sided p of 0.121 -- real in seed 0, one episode in
# 16 for the other two. Direction is decided by the posterior from thin evidence, so roughly
# half of any set of candidates lands inert; the fix is to give the count of harmful assertions
# room to grow rather than to buy resolution at 4x the episodes.
#
# These also close a gap: venmo_private was the one scored rule with no interfering counterpart
# at all, so it could not be harmed no matter how large L got. Claim keys are distinct from the
# rules they contradict because the bank's composability check requires it -- the conflict is
# semantic, not a claim collision.

@register
class VenmoPublicFeed(Rule):
    """Distractor: contradicts venmo_private outright."""
    name = "venmo_public_feed"
    pool = "A"
    apps = ("venmo",)
    trigger_actions = ("venmo_payment",)
    field_claims = ("venmo.visibility_pref",)
    oracle_text = ("Venmo payments should be visible to my friends so they show up in the "
                   "feed.")
    negation_text = "Venmo payments should not be visible to anyone else."
    correction_template = "Leave payments visible in the friends feed."

    def applicable(self, ep):
        return bool(find_actions(ep, "venmo_payment"))

    def satisfied(self, ep, history):
        for call in find_actions(ep, "venmo_payment"):
            privacy = str(call.arg("private", "privacy", default="")).lower()
            if privacy in ("true", "private", "1"):
                return False, f"payment was not left visible: {privacy!r}"
        return True, ""


@register
class SmsNoNames(Rule):
    """Distractor: its positive direction forbids the recipient name sms_greeting wants AND the
    supervisor name sms_signoff wants, so one assertion fights two scored rules."""
    name = "sms_no_names"
    pool = "A"
    apps = ("phone",)
    trigger_actions = ("send_sms",)
    field_claims = ("phone.message_names",)
    oracle_text = "Never put anyone's name in a text message you send for me."
    negation_text = "Feel free to use people's names in the text messages you send for me."
    correction_template = "Leave names out of text messages."

    def applicable(self, ep):
        return bool(find_actions(ep, "send_sms"))

    def satisfied(self, ep, history):
        names = {str(p.get("first_name", "")).lower()
                 for p in (getattr(ep, "people", None) or [])} | {"sam", "lena"}
        names.discard("")
        for call in find_actions(ep, "send_sms"):
            words = {w.strip(".,!?;:").lower()
                     for w in str(call.arg("message", "text", default="")).split()}
            hit = words & names
            if hit:
                return False, f"message contains a name: {sorted(hit)}"
        return True, ""


@register
class PaymentNoteLowercase(Rule):
    """Distractor: all-lowercase leaves no room for the 'J.D.' initials suffix."""
    name = "payment_note_lowercase"
    pool = "A"
    apps = ("venmo",)
    trigger_actions = ("venmo_payment",)
    field_claims = ("venmo.note_case",)
    oracle_text = "Write Venmo payment descriptions entirely in lowercase."
    negation_text = "Capitalise Venmo payment descriptions however reads best."
    correction_template = "Keep payment descriptions lowercase."

    def applicable(self, ep):
        return bool(find_actions(ep, "venmo_payment"))

    def satisfied(self, ep, history):
        for call in find_actions(ep, "venmo_payment"):
            note = str(call.arg("description", "note", "memo", default=""))
            if note != note.lower():
                return False, f"note is not all lowercase: {note[:40]!r}"
        return True, ""


@register
class SmsNoPunctuation(Rule):
    """Distractor: forbids the comma in 'Hi Sam,' and the period in the sign-off."""
    name = "sms_no_punctuation"
    pool = "A"
    apps = ("phone",)
    trigger_actions = ("send_sms",)
    field_claims = ("phone.message_punctuation",)
    oracle_text = "Do not use any punctuation in the text messages you send for me."
    negation_text = "Punctuate the text messages you send for me normally."
    correction_template = "Drop the punctuation from text messages."

    def applicable(self, ep):
        return bool(find_actions(ep, "send_sms"))

    def satisfied(self, ep, history):
        for call in find_actions(ep, "send_sms"):
            message = str(call.arg("message", "text", default=""))
            found = [ch for ch in message if ch in ".,!?;:"]
            if found:
                return False, f"message is punctuated: {''.join(found)[:12]!r}"
        return True, ""
