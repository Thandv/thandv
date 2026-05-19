from __future__ import annotations

import json


from thandv import session_promotion as sp


# --- Helpers --------------------------------------------------------------

def _write_session(thandv_home, events: list[dict], name: str = "test"):
    path = thandv_home / "sessions" / f"{name}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for ev in events:
            f.write(json.dumps(ev) + "\n")
    return path


# --- Pair extraction ------------------------------------------------------

def test_extract_pairs_simple_one_turn(thandv_home):
    path = _write_session(thandv_home, [
        {"role": "user", "content": "what is 7*8?"},
        {"role": "assistant", "content": "56"},
    ])
    pairs = sp.extract_pairs(path)
    assert pairs == [{"prompt": "what is 7*8?", "completion": "56"}]


def test_extract_pairs_multiple_turns(thandv_home):
    path = _write_session(thandv_home, [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "q2"},
        {"role": "assistant", "content": "a2"},
    ])
    pairs = sp.extract_pairs(path)
    assert pairs == [
        {"prompt": "q1", "completion": "a1"},
        {"prompt": "q2", "completion": "a2"},
    ]


def test_extract_pairs_strips_tool_blocks(thandv_home):
    """Tool-block bookkeeping must not appear in training pairs."""
    path = _write_session(thandv_home, [
        {"role": "user", "content": "read foo.py"},
        {
            "role": "assistant",
            "content": (
                "let me check.\n```tool\n{\"name\":\"read_file\",\"args\":{\"path\":\"foo.py\"}}\n```"
            ),
        },
        {"role": "tool", "name": "read_file", "result": {"content": "x = 1"}},
        {"role": "assistant", "content": "the file says `x = 1`."},
    ])
    pairs = sp.extract_pairs(path)
    # First assistant gets paired with the user (after stripping the tool block).
    # The "let me check." text remains; only the fenced ```tool``` block goes.
    assert len(pairs) == 1
    assert pairs[0]["prompt"] == "read foo.py"
    assert "```tool" not in pairs[0]["completion"]
    assert "let me check" in pairs[0]["completion"]


def test_extract_pairs_drops_tool_only_assistant(thandv_home):
    """An assistant turn that is ONLY a tool block has nothing to train on."""
    path = _write_session(thandv_home, [
        {"role": "user", "content": "list ."},
        {
            "role": "assistant",
            "content": "```tool\n{\"name\":\"list_dir\",\"args\":{\"path\":\".\"}}\n```",
        },
        {"role": "tool", "name": "list_dir", "result": {"entries": []}},
        {"role": "assistant", "content": "no entries."},
    ])
    pairs = sp.extract_pairs(path)
    # First assistant is tool-only → skipped. Final assistant has no preceding
    # user turn (it was consumed by the tool-only attempt? no — let's check
    # the actual semantics).
    # Our rule: last_user is reset after the first assistant. So the second
    # assistant has no user to pair with. Zero pairs.
    assert pairs == []


def test_extract_pairs_user_consumed_only_once(thandv_home):
    """Same user message should not pair with two assistant turns."""
    path = _write_session(thandv_home, [
        {"role": "user", "content": "only-prompt"},
        {"role": "assistant", "content": "first"},
        {"role": "assistant", "content": "second"},
    ])
    pairs = sp.extract_pairs(path)
    assert pairs == [{"prompt": "only-prompt", "completion": "first"}]


def test_extract_pairs_skips_tool_result_pseudo_user(thandv_home):
    """The legacy tool-result-as-user content must not become a prompt."""
    path = _write_session(thandv_home, [
        {"role": "user", "content": "do thing"},
        {"role": "assistant", "content": "x"},
        {"role": "user", "content": "```tool-result\n{}\n```"},   # legacy form
        {"role": "assistant", "content": "y"},
    ])
    pairs = sp.extract_pairs(path)
    # Only the first pair counts; the pseudo-user is ignored.
    assert pairs == [{"prompt": "do thing", "completion": "x"}]


def test_extract_pairs_ignores_leading_assistant(thandv_home):
    """An assistant event with no preceding user shouldn't crash or pair."""
    path = _write_session(thandv_home, [
        {"role": "assistant", "content": "(orphan)"},
        {"role": "user", "content": "real prompt"},
        {"role": "assistant", "content": "real answer"},
    ])
    pairs = sp.extract_pairs(path)
    assert pairs == [{"prompt": "real prompt", "completion": "real answer"}]


def test_extract_pairs_malformed_lines_skipped(thandv_home):
    path = _write_session(thandv_home, [{"role": "user", "content": "q"}])
    path.write_text(
        path.read_text() + "not valid json\n"
        + json.dumps({"role": "assistant", "content": "a"}) + "\n"
    )
    pairs = sp.extract_pairs(path)
    assert pairs == [{"prompt": "q", "completion": "a"}]


# --- Clean-session filter -------------------------------------------------

def test_session_is_clean_no_tool_errors(thandv_home):
    path = _write_session(thandv_home, [
        {"role": "user", "content": "q"},
        {"role": "tool", "name": "read_file", "result": {"content": "ok"}},
        {"role": "assistant", "content": "a"},
    ])
    assert sp.session_is_clean(path) is True


def test_session_is_clean_any_tool_error_dirties(thandv_home):
    path = _write_session(thandv_home, [
        {"role": "user", "content": "q"},
        {"role": "tool", "name": "read_file", "result": {"error": "no such file"}},
        {"role": "assistant", "content": "sorry"},
    ])
    assert sp.session_is_clean(path) is False


def test_session_is_clean_treats_no_tool_calls_as_clean(thandv_home):
    """A pure-chat session with no tools is automatically clean."""
    path = _write_session(thandv_home, [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ])
    assert sp.session_is_clean(path) is True


# --- Stats ----------------------------------------------------------------

def test_stats_basic_counts(thandv_home):
    path = _write_session(thandv_home, [
        {"role": "user", "content": "q1"},
        {"role": "tool", "name": "x", "result": {"ok": True}},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "q2"},
        {"role": "tool", "name": "y", "result": {"error": "oops"}},
        {"role": "assistant", "content": "a2"},
    ])
    s = sp.stats(path)
    assert s.n_user_turns == 2
    assert s.n_assistant_turns == 2
    assert s.n_tool_calls == 2
    assert s.n_tool_errors == 1
    assert s.is_clean is False
    assert s.n_pairs_extracted == 2


# --- Promotion ------------------------------------------------------------

def test_promote_session_writes_queue_and_provenance(thandv_home):
    path = _write_session(thandv_home, [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "q2"},
        {"role": "assistant", "content": "a2"},
    ])
    result = sp.promote_session(path)
    assert result is not None
    queue_path, provenance = result
    assert queue_path.exists()
    lines = queue_path.read_text().splitlines()
    assert len(lines) == 2

    prov_path = queue_path.with_suffix(queue_path.suffix + ".provenance.json")
    assert prov_path.exists()
    prov = json.loads(prov_path.read_text())
    assert prov["source"] == "session_promotion"
    assert prov["n_pairs"] == 2
    assert prov["stats"]["n_pairs_extracted"] == 2


def test_promote_session_returns_none_when_empty(thandv_home):
    """A session with no user-assistant pairs produces nothing."""
    path = _write_session(thandv_home, [
        {"role": "user", "content": "q"},
        {"role": "tool", "name": "x", "result": {}},
    ])
    assert sp.promote_session(path) is None


def test_promote_clean_sessions_skips_dirty(thandv_home):
    """Only sessions with zero tool errors get auto-promoted."""
    clean = _write_session(thandv_home, [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "a"},
    ], name="clean-one")
    dirty = _write_session(thandv_home, [
        {"role": "user", "content": "q"},
        {"role": "tool", "name": "x", "result": {"error": "boom"}},
        {"role": "assistant", "content": "sorry"},
    ], name="dirty-one")

    promoted = sp.promote_clean_sessions()
    assert len(promoted) == 1
    # The clean one was promoted; the dirty one wasn't.
    assert clean.with_suffix(clean.suffix + ".promoted").exists()
    assert not dirty.with_suffix(dirty.suffix + ".promoted").exists()


def test_promote_clean_sessions_idempotent(thandv_home):
    """Running twice doesn't double-promote — marker file gates it."""
    _write_session(thandv_home, [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "a"},
    ], name="once")
    first = sp.promote_clean_sessions()
    second = sp.promote_clean_sessions()
    assert len(first) == 1
    assert second == []


def test_promote_clean_sessions_max_sessions_honoured(thandv_home):
    for i in range(5):
        _write_session(thandv_home, [
            {"role": "user", "content": f"q{i}"},
            {"role": "assistant", "content": f"a{i}"},
        ], name=f"s{i}")
    promoted = sp.promote_clean_sessions(max_sessions=2)
    assert len(promoted) == 2


def test_promote_clean_sessions_handles_empty_session(thandv_home):
    """An empty clean session still gets its marker written so we don't
    re-scan it forever."""
    path = _write_session(thandv_home, [
        {"role": "user", "content": "q"},
        # No assistant reply.
    ], name="empty-one")
    promoted = sp.promote_clean_sessions()
    assert promoted == []
    assert path.with_suffix(path.suffix + ".promoted").exists()


# --- list_sessions --------------------------------------------------------

def test_list_sessions_orders_newest_first(thandv_home):
    import os
    import time
    older = _write_session(thandv_home, [{"role": "user", "content": "o"}], name="older")
    time.sleep(0.05)
    newer = _write_session(thandv_home, [{"role": "user", "content": "n"}], name="newer")
    # Force the order with mtime tweaks to be deterministic on fast hardware.
    os.utime(older, (older.stat().st_atime, older.stat().st_mtime - 5))
    sessions = sp.list_sessions()
    assert sessions[0] == newer
    assert older in sessions
