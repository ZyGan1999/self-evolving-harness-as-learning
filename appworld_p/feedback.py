"""User feedback channels.
Instance feedback describes observed violations without supplying oracle statements.
Corrective feedback supplies each violated rule's correction template.
Other tiers expose acceptance, affected applications, or preference aspects."""

from dataclasses import dataclass, field

from .rules import RuleResult


@dataclass
class Feedback:
    tier: str
    accepted: bool
    text: str = ""
    structured: dict[str, bool] = field(default_factory=dict)

    provenance: list[str] = field(default_factory=list)


def _complaint(result: RuleResult) -> str:
    """Instance-tier complaint for one violation, built only from the checker's detail.

    The detail strings are already instance-bound ("note lacks initials suffix: 'dinner'"),
    so the wrapper just makes them sound like a user rather than a linter. Rules whose
    detail is empty degrade to a bare complaint, which is the honest weaker signal.
    """
    if not result.detail:
        return "something about how you did this isn't right."
    return f"{result.detail} -- I don't want it done that way."


_ARTEFACT = {
    "send_sms": "the text message you sent",
    "venmo_payment": "the payment you made",
    "spotify_like_song": "the way you saved that music",
    "spotify_add_to_playlist": "the playlist you put that in",
    "spotify_create_playlist": "the playlist you made",
    "create_file": "the file you created",
    "create_note": "the note you wrote",
    "send_email": "the email you sent",
    "create_todoist_task": "the task you added",
}


_ASPECT = {
    "phone.message_prefix": "how you opened that text message",
    "phone.message_suffix": "how you ended that text message",
    "phone.message_punctuation": "the punctuation in that text message",
    "phone.message_length": "how long that text message was",
    "phone.message_names": "the names you put in that text message",
    "venmo.note": "what you wrote in the payment note",
    "venmo.note_prefix": "how the payment note started",
    "venmo.note_suffix": "how the payment note ended",
    "venmo.note_case": "the capitalisation in the payment note",
    "venmo.note_length": "how long the payment note was",
    "venmo.note_punctuation": "the punctuation in the payment note",
    "venmo.private": "who could see that payment",
    "venmo.payment_card": "which card you paid with",
    "spotify.save_action": "how you saved that music",
    "spotify.playlist_visibility": "who could see that playlist",
    "file_system.name": "what you named that file",
    "file_system.content_prefix": "how that file started",
}


def _aspects(violated: list[RuleResult], rules_by_name: dict | None) -> list[str]:
    """Aspect phrases for the violated rules, deduplicated, in first-appearance order.

    Unlike _scopes this does NOT collapse to one phrase per artefact: naming two aspects of the
    same message is the point of this tier. What it still withholds is the TARGET -- "how you
    ended that text message" says the ending is wrong without saying it should carry a first name,
    so the learner has a hypothesis space to search instead of a value to copy.
    """
    seen, out = set(), []
    for r in violated:
        rule = (rules_by_name or {}).get(r.rule)
        for claim in getattr(rule, "field_claims", ()) or ():
            phrase = _ASPECT.get(claim)
            if phrase and phrase not in seen:
                seen.add(phrase)
                out.append(phrase)
            break
    return out


def _scopes(violated: list[RuleResult], rules_by_name: dict | None) -> list[str]:
    """Distinct artefact phrases for the violated rules, in first-appearance order.

    Deduplicated on purpose: three broken sms rules must read as ONE complaint about the message,
    not three. Otherwise the count of complaint lines leaks how many rules were broken, which is
    attribution information this tier is defined not to carry.
    """
    seen, out = set(), []
    for r in violated:
        rule = (rules_by_name or {}).get(r.rule)
        for action in getattr(rule, "trigger_actions", ()) or ():
            phrase = _ARTEFACT.get(action)
            if phrase and phrase not in seen:
                seen.add(phrase)
                out.append(phrase)
            break
    return out


def make_feedback(results: list[RuleResult], tier: str = "default",
                  correction_templates: dict[str, str] | None = None,
                  rules_by_name: dict | None = None) -> Feedback:
    violated = [r for r in results if r.applicable and r.satisfied is False]
    accepted = not violated
    if tier == "aspect":
        if accepted:
            return Feedback(tier, True, "Thanks, that was done the way I like it.")
        parts = _aspects(violated, rules_by_name)
        if not parts:
            return Feedback(tier, False, "Hmm, I wasn't happy with how some of that was done.",
                            provenance=[r.rule for r in violated])
        listed = "\n".join(f"- {p}" for p in parts)

        text = ("Some things about how you did that aren't how I like them:\n"
                f"{listed}\n"
                "Keep doing the task the same way — just handle those parts differently. "
                "Don't ask me to approve things in advance; work it out and get it right.")
        return Feedback(tier, False, text, provenance=[r.rule for r in violated])
    if tier == "vague_scoped":
        if accepted:
            return Feedback(tier, True, "Thanks, that was done the way I like it.")
        scopes = _scopes(violated, rules_by_name)
        if not scopes:
            return Feedback(tier, False, "Hmm, I wasn't happy with how some of that was done.",
                            provenance=[r.rule for r in violated])
        if len(scopes) == 1:
            text = (f"I'm not happy with {scopes[0]} — that's not how I like it done. "
                    f"The action itself was fine, it's the way you wrote it.")
        else:
            joined = ", and ".join(scopes)
            text = (f"I'm not happy with {joined} — that's not how I like those done. "
                    f"The actions themselves were fine, it's the way you did them.")

        return Feedback(tier, False, text, provenance=[r.rule for r in violated])
    if tier == "default":
        text = ("Thanks, that was done the way I like it." if accepted
                else "Hmm, I wasn't happy with how some of that was done.")
        return Feedback(tier, accepted, text)
    if tier == "corrective":
        if accepted:
            return Feedback(tier, True, "Thanks, that was done the way I like it.")
        templates = correction_templates or {}
        lines = ["I wasn't happy with some of that:"]
        lines += [f"- {templates.get(r.rule, 'Please redo this differently.')}" for r in violated]
        return Feedback(tier, False, "\n".join(lines))
    if tier == "instance":
        if accepted:
            return Feedback(tier, True, "Thanks, that was done the way I like it.")

        lines = ["That's not how I like things done:"]
        lines += [f"- {_complaint(r)}" for r in violated]
        return Feedback(tier, False, "\n".join(lines), provenance=[r.rule for r in violated])
    if tier == "bypass":
        structured = {r.rule: bool(r.satisfied) for r in results if r.applicable}
        return Feedback(tier, accepted, "", structured)
    raise ValueError(f"unknown feedback tier: {tier}")
