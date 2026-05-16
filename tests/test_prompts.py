from thandv.prompts import SYSTEM_PROMPT


def test_system_prompt_mentions_tools_and_style():
    p = SYSTEM_PROMPT.lower()
    assert "tool" in p
    assert "terse" in p or "short" in p
    assert "thandv" in p
