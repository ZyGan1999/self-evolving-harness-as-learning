"""Weak feedback generation — the channel tiers (n_eff = iota * T knob).

Tier "default":      accept / generic dissatisfaction (lowest information)
Tier "vague_scoped": names the ARTEFACT that was wrong, never which rule or what to do
Tier "aspect":       names WHICH ASPECT of the artefact is wrong, never the target value
Tier "corrective":   per-violated-rule rewrite demonstration from correction_template
Tier "instance":     per-violated-rule complaint about THIS episode, no general rule stated
Tier "bypass":       structured per-rule verdicts, bypassing f's perception channel

Ordered by information content:
    default < vague_scoped < aspect < instance < corrective < bypass

"aspect" is the tier that separates ATTRIBUTION from the TARGET, which is what the two weaker
tiers could not do. Both of them failed by making the agent stop acting rather than by learning
more slowly: from bare dissatisfaction the learner inferred a permissions rule ("never
authenticate", "NEVER send messages"), and from an artefact-scoped complaint it inferred a
workflow rule ("preview exact wording BEFORE executing, get approval"). Both are meta-strategies,
not hypotheses about the artefact, and under the first the venmo denominator fell to zero.

aspect says "how you ended that text message" -- the learner knows the ending is the problem and
must still discover that the ending should carry a first name. That is a hypothesis space to
search rather than a value to copy, so partial learning is possible, which is the precondition for
a curve that descends and then stops.

"vague_scoped" exists because the gap between default and instance turned out to be the whole
experiment rather than one notch. Under `instance` a self-evolving learner drove violations from
0.745 to 0.062 over 48 episodes -- no ceiling at all. Under `default` it went the other way,
0.745 to 1.000, and the memory says why: from "I wasn't happy with how some of that was done" the
learner inferred a PRIVACY complaint and wrote fifteen lines of "never authenticate", "never send
messages without approval", "NEVER access credentials". The agent complied, stopped performing the
constrained actions, and every rule then read as violated. That is the model's prior taking over
an empty channel, not slower learning, so the two arms differ in topic as well as in precision and
neither isolates credit assignment.

vague_scoped keeps the topic fixed and removes only the attribution. It names the artefact the
user is unhappy with -- the text message, the payment note -- and says nothing about which of the
rules on that artefact was broken or how to fix it. The learner still has to decide among the
three sms rules or the two note rules, which is the credit-assignment problem, but it cannot
wander off into permissions. The same failure that motivated this tier appeared once in screening:
sms_terse's complaint literally reads "message longer than five words: 66" and the learner still
wrote "do not authenticate or log in using stored credentials" twice out of three times.

The "instance" tier exists because "corrective" hands the learner the finished rule.
correction_template is a fully quantified sentence ("Sign text messages with my first name
at the end."), so a learner that summarises it just copies it and no induction happens --
the assertions it writes are as good as oracle_text and redundancy can never help or hurt.
Real users complain about the case in front of them instead, which is what the checkers'
`detail` strings already are ("sms not signed with first name: 'Hi Alice, ...'"). From those
the learner has to guess the scope itself (all texts? only to Alice?), and that guess is
where imperfect memory comes from.
"""

from dataclasses import dataclass, field

from .rules import RuleResult


@dataclass
class Feedback:
    tier: str
    accepted: bool
    text: str = ""                                   # what the agent-side updater sees
    structured: dict[str, bool] = field(default_factory=dict)  # bypass tier only
    # Harness-only: which rule produced each complaint line, in the order they appear in
    # `text`. The learner never sees this; it is how a generated assertion gets labelled
    # with the rule it describes without having to classify free text after the fact.
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


# Which artefact each constrained action produces, in the words a user would use. The scope hint
# is built from the rule's own trigger_actions, so a new rule inherits a hint from its action
# rather than needing one authored here -- and an action with no entry degrades to the bare
# complaint instead of silently naming the wrong artefact.
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


# Which ASPECT of the artefact each field_claim refers to, again in a user's words. field_claims
# already encodes exactly this distinction ('phone.message_suffix' vs 'phone.message_punctuation'),
# so the aspect tier reads it rather than introducing a parallel per-rule mapping that could drift
# out of step with the checkers.
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
            break          # a rule's first field claim names its aspect
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
            break          # a rule's first trigger action names its artefact
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
        # "Keep doing the task" is stated because both weaker tiers failed by making the agent
        # stop acting: default inferred a permissions rule, vague_scoped inferred a
        # preview-and-ask-approval workflow. Neither is a hypothesis about the artefact.
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
            # No artefact could be named, so this degrades to the default tier's text rather
            # than inventing a scope. Honest weaker signal, same policy as _complaint().
            return Feedback(tier, False, "Hmm, I wasn't happy with how some of that was done.",
                            provenance=[r.rule for r in violated])
        if len(scopes) == 1:
            text = (f"I'm not happy with {scopes[0]} — that's not how I like it done. "
                    f"The action itself was fine, it's the way you wrote it.")
        else:
            joined = ", and ".join(scopes)
            text = (f"I'm not happy with {joined} — that's not how I like those done. "
                    f"The actions themselves were fine, it's the way you did them.")
        # provenance is harness-only (never shown to the learner) and is what lets the analysis
        # ask which rule a complaint was ABOUT while the learner had to guess.
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
        # One uniform wrapper over every checker's own detail string -- no per-rule
        # authoring, so the noise level is a property of the environment rather than
        # something tuned per rule.
        lines = ["That's not how I like things done:"]
        lines += [f"- {_complaint(r)}" for r in violated]
        return Feedback(tier, False, "\n".join(lines), provenance=[r.rule for r in violated])
    if tier == "bypass":
        structured = {r.rule: bool(r.satisfied) for r in results if r.applicable}
        return Feedback(tier, accepted, "", structured)
    raise ValueError(f"unknown feedback tier: {tier}")
