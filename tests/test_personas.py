import pytest

from thandv.personas import (
    CODE,
    FINANCE,
    PERSONAS,
    WRITER,
    get_persona,
    list_personas,
)


def test_three_personas_registered():
    assert set(PERSONAS) == {"code", "writer", "finance"}
    assert list_personas() == ["code", "finance", "writer"]


def test_get_persona_returns_singleton():
    assert get_persona("code") is CODE
    assert get_persona("writer") is WRITER
    assert get_persona("finance") is FINANCE


def test_get_persona_unknown_raises():
    with pytest.raises(ValueError, match="unknown persona"):
        get_persona("astrology")


def test_each_persona_has_required_fields():
    for p in PERSONAS.values():
        assert p.name
        assert p.description
        assert p.system_prompt
        assert p.skills
        assert "thandv" in p.system_prompt.lower()


def test_finance_has_disclaimer_and_warns_explicitly():
    assert FINANCE.disclaimer
    prompt = FINANCE.system_prompt.lower()
    assert "not investment advice" in prompt
    assert "do not" in prompt
    assert "no edge" in prompt or "no real-time" in prompt


def test_finance_refuses_recommend_language():
    # Tripwire: if someone softens the finance prompt later, this fires.
    prompt = FINANCE.system_prompt
    assert "Recommend specific securities" in prompt or "recommend specific securities" in prompt
    assert "Predict market direction" in prompt or "predict market direction" in prompt


def test_writer_prompt_mentions_voice_and_editing():
    p = WRITER.system_prompt.lower()
    assert "voice" in p
    assert "edit" in p


def test_code_prompt_mentions_tools_and_safety():
    p = CODE.system_prompt.lower()
    assert "tool" in p
    assert "destructive" in p or "rm -rf" in p


def test_all_personas_share_honesty_skill():
    for p in PERSONAS.values():
        assert "honesty" in p.skills, f"{p.name} should include honesty skill"
