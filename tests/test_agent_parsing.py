"""extract_code must handle fenced blocks, pseudo-XML tool calls, and raw text."""

from appworld_p.agent import extract_code


def test_fenced_block():
    text = "Let me look.\n```python\nprint(1)\n```\ndone"
    assert extract_code(text) == "print(1)"


def test_first_fenced_block_wins():
    # first block = this turn's action; later blocks are hallucinated rollouts
    text = "```python\nprint(1)\n```\nthen\n```python\nprint(2)\n```"
    assert extract_code(text) == "print(1)"


def test_hallucinated_rollout_defused():
    text = (
        "```python\nprint(apis.api_docs.show_api_descriptions(app_name='phone'))\n```\n"
        "```\nfake output pretending to be the environment\n```\n"
        "```python\nresult = apis.p\n```\n"
        "Task marked as complete.\n"
    )
    assert extract_code(text) == \
        "print(apis.api_docs.show_api_descriptions(app_name='phone'))"


def test_pseudo_xml_tool_calls():
    text = (
        "I'll help you.\n"
        '<function_calls>\n<invoke name="function_call">\n'
        '<parameter name="function_name">function_call</parameter>\n'
        '<parameter name="code">\nprint(apis.api_docs.show_app_descriptions())\n</parameter>\n'
        "</invoke>\n</function_calls>\n"
        '<function_calls>\n<invoke name="function_call">\n'
        '<parameter name="code">\ncreds = apis.supervisor.show_account_passwords()\n</parameter>\n'
        "</invoke>\n</function_calls>"
    )
    code = extract_code(text)
    assert "show_app_descriptions" in code          # first pseudo-XML block executed
    assert "show_account_passwords" not in code     # later blocks = same-turn rollout, dropped
    assert "<parameter" not in code and "<invoke" not in code


def test_raw_text_fallback():
    assert extract_code("print(3)") == "print(3)"


# ---- FunctionCallAgent protocol validator ---------------------------------

from appworld_p.agent import validate_api_call  # noqa: E402


def test_fc_accepts_literal_call():
    code, err = validate_api_call('apis.venmo.create_transaction(receiver="a@b.com", '
                                  'amount=74.5, private=True, payment_card_id=97)')
    assert err == "" and code.startswith("apis.venmo.create_transaction(")


def test_fc_accepts_nested_literals_and_negatives():
    code, err = validate_api_call("apis.foo.bar(xs=[1, -2, 3], d={'k': 'v'}, n=None)")
    assert err == "" and code is not None


def test_fc_rejects_variable_args():
    code, err = validate_api_call("apis.venmo.create_transaction(amount=amount)")
    assert code is None and "literal" in err


def test_fc_rejects_expressions():
    code, err = validate_api_call("apis.phone.send_text_message(message='hi ' + name)")
    assert code is None
    code, err = validate_api_call("apis.phone.send_text_message(message=f'{x}')")
    assert code is None


def test_fc_rejects_multiple_statements():
    code, err = validate_api_call("x = 1\napis.foo.bar(a=1)")
    assert code is None
    code, err = validate_api_call("apis.foo.bar(a=1); apis.foo.baz(b=2)")
    assert code is None


def test_fc_rejects_non_api_calls():
    assert validate_api_call("print(1)")[0] is None
    assert validate_api_call("len([1,2])")[0] is None
    assert validate_api_call("apis.foo(a=1)")[0] is None       # too-short chain
    assert validate_api_call("evil.apis.foo.bar(a=1)")[0] is None


def test_fc_rejects_loops_and_imports():
    assert validate_api_call("for i in range(3): apis.foo.bar(a=i)")[0] is None
    assert validate_api_call("import os")[0] is None


def test_fc_rejects_call_args():
    # a call nested as an argument is an expression, not a literal
    assert validate_api_call("apis.foo.bar(a=apis.baz.qux())")[0] is None


def test_fc_canonicalizes_json_booleans():
    code, err = validate_api_call('apis.venmo.create_transaction(private=true, x=false, y=null)')
    assert err == ""
    assert "private=True" in code and "x=False" in code and "y=None" in code
