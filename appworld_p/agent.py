"""Agents that solve AppWorld tasks.

ReactCodeAgent:    minimal ReAct code agent (LLM writes python against world.execute).
FunctionCallAgent: constrained actuator — one API call per turn, literal args only
                   (AST-enforced). The convex-hull claims are honest on this agent:
                   f cannot outsource computation to the interpreter (Q1 redesign,
                   see refine-logs/Q1_REDESIGN.md).
OracleAgent:       executes the official ground-truth solution (train/dev only) — the
                   cheap execution mode for pipeline validation and Exp-1 sensitivity.

Harness injection: `memory` (context class) is inserted into the system prompt;
`external_stats` (control class, external counter) is injected as a structured
section.
"""

import ast
import re
from typing import Any

from .llm import BaseLLM

SYSTEM_PROMPT = """\
You are an autonomous assistant that completes day-to-day tasks for a user (the \
"supervisor") by writing Python code executed in a stateful REPL with access to app APIs.

How to operate:
- Respond with exactly ONE fenced python code block per turn (```python ... ```). The \
environment executes it and returns the output; only what you print() is shown to you.
- NEVER use XML/tool-call syntax like <function_calls> or <invoke> — this environment \
only understands a single fenced python block. Thoughts go in # comments.
- STOP after your code block. Never write the execution output yourself, and never \
continue with further steps in the same reply — you will receive the REAL output next turn. \
Only the FIRST code block of your reply is executed.
- Do not call apis.supervisor.complete_task() until you have actually finished the work \
(or are reporting failure with status="fail") — completing with nothing done fails the task.
- Write small chunks of code, one step at a time. Verify things work before making \
irreversible changes. Variables persist across turns.
- Call APIs as: apis.{{app_name}}.{{api_name}}(**kwargs). Apps: {app_names}.
- Explore documentation whenever unsure — this is always the right first move:
    print(apis.api_docs.show_app_descriptions())
    print(apis.api_docs.show_api_descriptions(app_name='...'))
    print(apis.api_docs.show_api_doc(app_name='...', api_name='...'))

Key rules (violating these fails the task):
1. All the supervisor's personal data, account passwords, addresses, and payment cards \
live in the "supervisor" app (e.g., apis.supervisor.show_account_passwords()). Log in to \
other apps with those credentials to get access tokens.
2. APIs that return results in "pages" must be iterated over ALL pages (page_index=...).
3. Any reference to friends, family, or other people refers to the phone app's contacts. \
Any reference to a "file system" means the file_system app, not the OS.
4. For current date/time use datetime.now() or the phone app — never your own knowledge. \
For "yesterday" etc. use full day boundaries 00:00:00-23:59:59.
5. Use only the provided APIs and the Python standard library (no external packages; \
OS-affecting modules are disabled).
6. When done, call apis.supervisor.complete_task(). If the task asks a question, pass \
answer=<answer> where the answer is ONLY the bare entity or number — no full sentences, \
no units, no currency symbols, no formatting (e.g. answer=275, not "$275.00"; numbers \
as digits, not words). For yes/no questions, answer "yes" or "no". If the task requires \
no answer, skip the argument. If you are certain you cannot solve it, call \
apis.supervisor.complete_task(status="fail").
7. Decide everything autonomously; never ask for clarification.
8. If an API call fails with "No API named ...", do NOT guess another name — immediately \
print(apis.api_docs.show_api_descriptions(app_name='...')) and pick from the real list.
{memory_section}{stats_section}"""

MEMORY_TEMPLATE = "\nWhat you remember about this user:\n<memory>\n{memory}\n</memory>\n"
STATS_TEMPLATE = "\nLive statistics about this user (maintained externally):\n<stats>\n{stats}\n</stats>\n"

CODE_RE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)
# Some models/relays emit pseudo-XML tool-call syntax instead of fenced blocks.
XML_CODE_RE = re.compile(r'<parameter name="code">\s*(.*?)\s*</parameter>', re.DOTALL)


def extract_code(text: str) -> str:
    """Take the FIRST fenced block: in the one-block-per-turn protocol the first
    block is this turn's action — anything after it is a hallucinated rollout
    (models sometimes fabricate execution outputs and further steps)."""
    blocks = CODE_RE.findall(text)
    if blocks:
        return blocks[0].strip()
    xml_blocks = XML_CODE_RE.findall(text)
    if xml_blocks:
        return xml_blocks[0].strip()
    return text.strip()


class ReactCodeAgent:
    # No autofill hook: this agent writes free-form python, so there is no single validated call
    # to intercept. The autofill harness is defined only for the constrained FC actuator.
    def __init__(self, llm: BaseLLM, max_steps: int = 30, memory: str = "",
                 external_stats: str = "", temperature: float = 0.7,
                 verbose: bool = False):
        self.llm = llm
        self.max_steps = max_steps
        self.memory = memory
        self.external_stats = external_stats
        self.temperature = temperature
        self.verbose = verbose

    def solve(self, world: Any) -> dict:
        app_names = ", ".join(sorted(world.task.app_descriptions.keys())) \
            if hasattr(world.task, "app_descriptions") else "(see api_docs)"
        system = SYSTEM_PROMPT.format(
            app_names=app_names,
            memory_section=MEMORY_TEMPLATE.format(memory=self.memory) if self.memory else "",
            stats_section=STATS_TEMPLATE.format(stats=self.external_stats)
            if self.external_stats else "",
        )
        supervisor = dict(world.task.supervisor)
        messages: list[dict] = [{
            "role": "user",
            "content": (f"Task from {supervisor.get('first_name', '')} "
                        f"{supervisor.get('last_name', '')} "
                        f"(email: {supervisor.get('email', '')}, "
                        f"phone: {supervisor.get('phone_number', '')}):\n\n"
                        f"{world.task.instruction}\n\nStart now."),
        }]
        steps = 0
        transcript: list[dict] = []
        for _ in range(self.max_steps):
            steps += 1
            reply = self.llm.generate(system, messages, temperature=self.temperature)
            code = extract_code(reply)
            output = world.execute(code)
            messages.append({"role": "assistant", "content": reply})
            messages.append({"role": "user", "content": f"Output:\n{str(output)[:4000]}"})
            transcript.append({"step": steps, "raw_reply": reply, "code": code,
                               "output": str(output)[:2000]})
            if self.verbose:
                print(f"      step {steps}/{self.max_steps}: "
                      f"{code.strip().splitlines()[0][:70] if code.strip() else '(empty)'}",
                      flush=True)
            if world.task_completed():
                break
        return {"steps": steps, "transcript_messages": len(messages),
                "transcript": transcript, "system_prompt": system}


FC_SYSTEM_PROMPT = """\
You are an autonomous assistant that completes day-to-day tasks for a user (the \
"supervisor") by calling app APIs, one call at a time.

STRICT PROTOCOL (enforced by a parser — violations are rejected, the step is wasted):
- Respond with exactly ONE fenced code block containing exactly ONE API call:
    ```
    apis.{{app_name}}.{{api_name}}(param1="value", param2=123)
    ```
- Every argument must be a LITERAL value (string, number, true/false, list/dict of \
literals). You cannot use variables, arithmetic, string operations, loops, imports, or \
any other Python — this environment is NOT a programming environment. Copy any value \
you need (access tokens, ids, phone numbers) literally from earlier outputs.
- Any reasoning goes in plain text BEFORE the code block, never inside it.
- STOP after the code block. You will receive the call's real output next turn. Never \
fabricate outputs or write further steps in the same reply.
- Apps available: {app_names}.
- Explore documentation whenever unsure — this is always the right first move:
    apis.api_docs.show_app_descriptions()
    apis.api_docs.show_api_descriptions(app_name='...')
    apis.api_docs.show_api_doc(app_name='...', api_name='...')

Key rules (violating these fails the task):
1. All the supervisor's personal data, account passwords, addresses, and payment cards \
live in the "supervisor" app (e.g., apis.supervisor.show_account_passwords()). Log in to \
other apps with those credentials to get access tokens.
2. APIs that return results in "pages" must be checked over ALL pages — make one call \
per page_index until a page comes back empty.
3. Any reference to friends, family, or other people refers to the phone app's contacts. \
Any reference to a "file system" means the file_system app, not the OS.
4. When done, call apis.supervisor.complete_task(). If the task asks a question, pass \
answer=<answer> where the answer is ONLY the bare entity or number — no full sentences, \
no units, no currency symbols (e.g. answer=275). For yes/no questions, answer "yes" or \
"no". If the task requires no answer, skip the argument. If you are certain you cannot \
solve it, call apis.supervisor.complete_task(status="fail").
5. Decide everything autonomously; never ask for clarification.
6. If an API call fails with "No API named ...", do NOT guess another name — immediately \
call apis.api_docs.show_api_descriptions(app_name='...') and pick from the real list.
{memory_section}{stats_section}"""

FC_PROTOCOL_REMINDER = ('Respond with exactly one API call in a fenced block, e.g. '
                        '```\napis.phone.send_text_message(phone_number="...", '
                        'message="...", access_token="...")\n```')


def _is_literal(node: ast.expr) -> bool:
    try:
        ast.literal_eval(node)
        return True
    except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
        return False


_JSONISH = {"true": True, "false": False, "null": None}


class _JsonLiteralFixer(ast.NodeTransformer):
    """Models habitually write JSON booleans (private=true); canonicalize them —
    they are still literals, just in the wrong dialect."""

    def visit_Name(self, node: ast.Name):
        if node.id in _JSONISH:
            return ast.copy_location(ast.Constant(_JSONISH[node.id]), node)
        return node


def validate_api_call(code: str) -> tuple[str | None, str]:
    """Validate the constrained-actuator protocol: exactly one expression statement of
    the form apis.<app>.<api>(<literal args>). Returns (canonical_code, error)."""
    try:
        tree = ast.parse(code.strip())
    except SyntaxError as exc:
        return None, f"not parseable ({exc.msg})"
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.Expr):
        return None, "must be exactly one API-call expression, nothing else"
    call = tree.body[0].value
    if not isinstance(call, ast.Call):
        return None, "must be an API call"
    call = ast.fix_missing_locations(_JsonLiteralFixer().visit(call))
    func = call.func
    if not (isinstance(func, ast.Attribute)
            and isinstance(func.value, ast.Attribute)
            and isinstance(func.value.value, ast.Name)
            and func.value.value.id == "apis"):
        return None, "call must be of the form apis.<app>.<api>(...)"
    for node in list(call.args) + [kw.value for kw in call.keywords]:
        if not _is_literal(node):
            return None, ("arguments must be literal values only — no variables, "
                          "arithmetic, or expressions; copy concrete values from "
                          "earlier outputs")
    return ast.unparse(call), ""


class FunctionCallAgent:
    """Constrained actuator: one literal-args API call per turn (see module docstring)."""

    def __init__(self, llm: BaseLLM, max_steps: int = 30, memory: str = "",
                 external_stats: str = "", temperature: float = 0.7,
                 verbose: bool = False, autofill=None):
        self.llm = llm
        self.max_steps = max_steps
        self.memory = memory
        self.external_stats = external_stats
        self.temperature = temperature
        self.verbose = verbose
        self.autofill = autofill

    def solve(self, world: Any) -> dict:
        app_names = ", ".join(sorted(world.task.app_descriptions.keys())) \
            if hasattr(world.task, "app_descriptions") else "(see api_docs)"
        system = FC_SYSTEM_PROMPT.format(
            app_names=app_names,
            memory_section=MEMORY_TEMPLATE.format(memory=self.memory) if self.memory else "",
            stats_section=STATS_TEMPLATE.format(stats=self.external_stats)
            if self.external_stats else "",
        )
        supervisor = dict(world.task.supervisor)
        messages: list[dict] = [{
            "role": "user",
            "content": (f"Task from {supervisor.get('first_name', '')} "
                        f"{supervisor.get('last_name', '')} "
                        f"(email: {supervisor.get('email', '')}, "
                        f"phone: {supervisor.get('phone_number', '')}):\n\n"
                        f"{world.task.instruction}\n\nStart now."),
        }]
        steps = 0
        rejected = 0
        transcript: list[dict] = []
        autofill_log: list[dict] = []
        for _ in range(self.max_steps):
            steps += 1
            reply = self.llm.generate(system, messages, temperature=self.temperature)
            code = extract_code(reply)
            canonical, error = validate_api_call(code)
            if canonical is None:
                rejected += 1
                output = f"PROTOCOL ERROR: {error}. {FC_PROTOCOL_REMINDER}"
            else:
                # control-class harness: rewrite the preference-owned field from the harness's
                # own ledger before the call lands. Placed after validation so a malformed call
                # still surfaces as a protocol error rather than being silently repaired.
                if self.autofill is not None:
                    canonical, repair = self.autofill.apply(canonical)
                    if repair and repair["changed"]:
                        autofill_log.append({"step": steps, **repair})
                output = world.execute(f"print({canonical})")
            messages.append({"role": "assistant", "content": reply})
            messages.append({"role": "user", "content": f"Output:\n{str(output)[:4000]}"})
            transcript.append({"step": steps, "raw_reply": reply,
                               "code": canonical or code, "rejected": canonical is None,
                               "output": str(output)[:2000]})
            if self.verbose:
                label = canonical or f"REJECTED: {code.strip().splitlines()[0][:50] if code.strip() else '(empty)'}"
                print(f"      step {steps}/{self.max_steps}: {label[:70]}", flush=True)
            if world.task_completed():
                break
        return {"steps": steps, "rejected_steps": rejected,
                "transcript_messages": len(messages),
                "transcript": transcript, "system_prompt": system,
                "autofill_repairs": autofill_log}


class OracleAgent:
    """Executes the official validation solution (requires ground_truth_mode='full')."""

    def solve(self, world: Any) -> dict:
        ground_truth = world.task.ground_truth
        code = ground_truth.compiled_solution_code + "\nsolution(apis, requester)"
        world.execute(code)
        return {"steps": 1, "oracle": True}
