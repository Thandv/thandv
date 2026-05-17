from __future__ import annotations

import json


from thandv.agent import Agent
from thandv.config import Config


def _agent(thandv_home, monkeypatch, replies: list[str]) -> Agent:
    """Build an Agent whose `_raw_stream` yields each canned reply as one chunk."""
    agent = Agent(config=Config(model="fake-model"))
    reply_iter = iter(replies)

    def fake_stream():
        return iter([next(reply_iter)])

    monkeypatch.setattr(agent, "_raw_stream", fake_stream)
    return agent


def _agent_chunked(thandv_home, monkeypatch, chunk_lists: list[list[str]]) -> Agent:
    """Build an Agent whose `_raw_stream` yields the *given* chunks on each call.

    Each entry in `chunk_lists` is a list of chunks for one model turn.
    """
    agent = Agent(config=Config(model="fake-model"))
    list_iter = iter(chunk_lists)

    def fake_stream():
        return iter(next(list_iter))

    monkeypatch.setattr(agent, "_raw_stream", fake_stream)
    return agent


# --- tool-call extraction ---------------------------------------------------

def test_extract_valid():
    text = 'sure, reading the file.\n```tool\n{"name": "read_file", "args": {"path": "x.py"}}\n```'
    call = Agent._extract_tool_call(text)
    assert call == {"name": "read_file", "args": {"path": "x.py"}}


def test_extract_no_block():
    assert Agent._extract_tool_call("just a chat reply") is None


def test_extract_malformed_json():
    text = "```tool\n{not json}\n```"
    assert Agent._extract_tool_call(text) is None


def test_extract_missing_name():
    text = '```tool\n{"args": {}}\n```'
    assert Agent._extract_tool_call(text) is None


def test_extract_args_default_empty():
    text = '```tool\n{"name": "list_dir"}\n```'
    call = Agent._extract_tool_call(text)
    assert call == {"name": "list_dir", "args": {}}


def test_extract_first_block_wins():
    text = (
        "```tool\n"
        '{"name": "a", "args": {}}\n'
        "```\n"
        "```tool\n"
        '{"name": "b", "args": {}}\n'
        "```"
    )
    call = Agent._extract_tool_call(text)
    assert call["name"] == "a"


# --- full turn loop ---------------------------------------------------------

def test_turn_plain_reply(thandv_home, monkeypatch):
    agent = _agent(thandv_home, monkeypatch, ["56"])
    out = "".join(agent.turn("what is 7*8?"))
    assert out.strip() == "56"
    assert agent.messages[-1] == {"role": "assistant", "content": "56"}


def test_turn_tool_call_then_reply(thandv_home, monkeypatch, tmp_path):
    target = tmp_path / "hi.txt"
    target.write_text("hello")
    replies = [
        f'reading it.\n```tool\n{{"name": "read_file", "args": {{"path": "{target}"}}}}\n```',
        "the file says hello.",
    ]
    agent = _agent(thandv_home, monkeypatch, replies)
    out = "".join(agent.turn("read hi.txt"))
    assert "the file says hello" in out
    assert "[tool] read_file" in out
    # The raw tool-block JSON must not leak into the user-visible stream.
    assert '"name": "read_file"' not in out
    assert "```tool\n" not in out
    roles = [m["role"] for m in agent.messages]
    assert roles == ["system", "user", "assistant", "user", "assistant"]
    assert "tool-result" in agent.messages[-2]["content"]


def test_turn_session_logged(thandv_home, monkeypatch):
    agent = _agent(thandv_home, monkeypatch, ["ok"])
    list(agent.turn("hi"))
    lines = agent.session_path.read_text().splitlines()
    events = [json.loads(line) for line in lines]
    assert events[0] == {"role": "user", "content": "hi"}
    assert events[1] == {"role": "assistant", "content": "ok"}


def test_turn_hop_budget(thandv_home, monkeypatch):
    tool_reply = '```tool\n{"name": "list_dir", "args": {"path": "."}}\n```'
    agent = _agent(thandv_home, monkeypatch, [tool_reply] * 100)
    out = "".join(agent.turn("loop forever"))
    assert "tool-call budget exhausted" in out


def test_system_prompt_includes_persona_skills_and_memory(thandv_home, monkeypatch):
    # The code persona requires the "coding-style" skill, so writing it here
    # ensures it gets loaded into the system prompt. A skill not listed in
    # the persona should be filtered out.
    (thandv_home / "skills" / "coding-style.md").write_text("be terse")
    (thandv_home / "skills" / "off-topic.md").write_text("ignore me")
    (thandv_home / "memory" / "fact.md").write_text("user likes python")
    agent = _agent(thandv_home, monkeypatch, ["k"])
    system = agent.messages[0]["content"]
    assert "be terse" in system
    assert "ignore me" not in system
    assert "user likes python" in system
    # Persona-specific prompt content
    assert "code persona" in system.lower() or "coding assistant" in system.lower()


def test_agent_uses_persona_from_config(thandv_home, monkeypatch):
    cfg = Config(model="fake-model", persona="writer")
    agent = Agent(config=cfg)
    monkeypatch.setattr(agent, "_raw_stream", lambda: iter(["ok"]))
    assert agent.persona.name == "writer"
    assert "writer persona" in agent.messages[0]["content"].lower()


def test_agent_explicit_persona_overrides_config(thandv_home, monkeypatch):
    from thandv.personas import get_persona

    cfg = Config(model="fake-model", persona="code")
    agent = Agent(config=cfg, persona=get_persona("finance"))
    monkeypatch.setattr(agent, "_raw_stream", lambda: iter(["ok"]))
    assert agent.persona.name == "finance"
    assert "finance persona" in agent.messages[0]["content"].lower()
    assert "not investment advice" in agent.messages[0]["content"].lower()


def test_agent_injects_persona_into_retrieve(thandv_home, monkeypatch):
    """When the model emits a retrieve tool call without an explicit persona,
    the agent fills in the active persona before dispatch."""
    from thandv import tools

    captured: dict = {}

    def fake_retrieve(**kwargs):
        captured.update(kwargs)
        return {"results": []}

    monkeypatch.setitem(tools.TOOLS, "retrieve", fake_retrieve)

    replies = [
        '```tool\n{"name": "retrieve", "args": {"query": "x", "k": 3}}\n```',
        "ok",
    ]
    agent = _agent(thandv_home, monkeypatch, replies)
    list(agent.turn("look it up"))
    assert captured["persona"] == "code"
    assert captured["query"] == "x"


def test_agent_retrieve_persona_override_respected(thandv_home, monkeypatch):
    from thandv import tools

    captured: dict = {}

    def fake_retrieve(**kwargs):
        captured.update(kwargs)
        return {"results": []}

    monkeypatch.setitem(tools.TOOLS, "retrieve", fake_retrieve)

    replies = [
        '```tool\n{"name": "retrieve", "args": {"query": "x", "persona": "all", "k": 2}}\n```',
        "ok",
    ]
    agent = _agent(thandv_home, monkeypatch, replies)
    list(agent.turn("look across all"))
    assert captured["persona"] == "all"


# --- streaming behaviour ----------------------------------------------------

def test_turn_streams_chunks_in_order(thandv_home, monkeypatch):
    chunks = ["hel", "lo ", "world"]
    agent = _agent_chunked(thandv_home, monkeypatch, [chunks])
    yielded = list(agent.turn("hi"))
    # Concatenation must equal the full reply, and chunks should be emitted
    # in roughly the same shape (not buffered into a single blob).
    assert "".join(yielded).strip() == "hello world"
    assert len(yielded) >= 2  # streamed, not collapsed


def test_turn_hides_tool_block_when_marker_spans_chunks(thandv_home, monkeypatch, tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("contents")
    # The "```tool\n" marker is intentionally split awkwardly across chunks.
    tool_call_chunks = [
        "sure, reading.\n",
        "``",
        "`to",
        "ol\n",
        '{"name": "read_file", "args": {"path": "',
        str(f),
        '"}}',
        "\n```",
    ]
    final_chunks = ["the file ", "says contents."]
    agent = _agent_chunked(thandv_home, monkeypatch, [tool_call_chunks, final_chunks])
    out = "".join(agent.turn("read it"))
    assert "sure, reading." in out
    assert "the file says contents." in out
    assert "```tool" not in out
    assert '"name": "read_file"' not in out


def test_turn_handles_tool_block_at_very_start(thandv_home, monkeypatch):
    # No preamble — model opens with the tool block immediately.
    chunks_call = ['```tool\n{"name": "list_dir", "args": {"path": "."}}\n```']
    chunks_final = ["done."]
    agent = _agent_chunked(thandv_home, monkeypatch, [chunks_call, chunks_final])
    out = "".join(agent.turn("list"))
    assert "[tool] list_dir" in out
    assert "done." in out
    assert "```tool" not in out
