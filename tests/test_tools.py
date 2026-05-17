
from thandv import tools
from thandv.tools import dispatch, edit_file, list_dir, read_file, retrieve, run_bash, write_file


def test_read_file_ok(tmp_path):
    p = tmp_path / "hello.txt"
    p.write_text("hi there")
    out = read_file(str(p))
    assert out["content"] == "hi there"
    assert out["path"].endswith("hello.txt")


def test_read_file_missing(tmp_path):
    out = read_file(str(tmp_path / "nope.txt"))
    assert "error" in out and "not found" in out["error"]


def test_read_file_is_directory(tmp_path):
    out = read_file(str(tmp_path))
    assert "error" in out and "directory" in out["error"]


def test_read_file_truncates_large(tmp_path, monkeypatch):
    monkeypatch.setattr(tools, "MAX_READ_BYTES", 10)
    p = tmp_path / "big.txt"
    p.write_text("abcdefghijklmnop")
    out = read_file(str(p))
    assert out["content"] == "abcdefghij"


def test_write_file_creates_parents(tmp_path):
    target = tmp_path / "a" / "b" / "c.txt"
    out = write_file(str(target), "yo")
    assert out["bytes"] == 2
    assert target.read_text() == "yo"


def test_edit_file_ok(tmp_path):
    p = tmp_path / "x.py"
    p.write_text("def foo(): pass\n")
    out = edit_file(str(p), "foo", "bar")
    assert out["ok"] is True
    assert p.read_text() == "def bar(): pass\n"


def test_edit_file_old_not_found(tmp_path):
    p = tmp_path / "x.py"
    p.write_text("def foo(): pass\n")
    out = edit_file(str(p), "zzz", "qqq")
    assert "error" in out


def test_edit_file_old_not_unique(tmp_path):
    p = tmp_path / "x.py"
    p.write_text("foo foo")
    out = edit_file(str(p), "foo", "bar")
    assert "error" in out and "unique" in out["error"]


def test_edit_file_missing(tmp_path):
    out = edit_file(str(tmp_path / "nope.py"), "a", "b")
    assert "error" in out and "not found" in out["error"]


def test_list_dir(tmp_path):
    (tmp_path / "a.txt").write_text("a")
    (tmp_path / "sub").mkdir()
    out = list_dir(str(tmp_path))
    names = {e["name"]: e["type"] for e in out["entries"]}
    assert names["a.txt"] == "file"
    assert names["sub"] == "dir"


def test_list_dir_not_a_directory(tmp_path):
    p = tmp_path / "a.txt"
    p.write_text("x")
    out = list_dir(str(p))
    assert "error" in out


def test_run_bash_success():
    out = run_bash("echo hello")
    assert out["exit"] == 0
    assert "hello" in out["stdout"]


def test_run_bash_nonzero_exit():
    out = run_bash("false")
    assert out["exit"] != 0


def test_run_bash_refuses_dangerous():
    out = run_bash("rm -rf /tmp/definitely-does-not-exist-xyz")
    assert "error" in out
    assert "destructive" in out["error"]


def test_run_bash_dangerous_with_confirm(tmp_path):
    # `rm -rf` on a tmp path *is* destructive but should be allowed when
    # the caller explicitly confirms.
    victim = tmp_path / "doomed"
    victim.mkdir()
    (victim / "f").write_text("x")
    out = run_bash(f"rm -rf {victim}", confirm=True)
    assert out["exit"] == 0
    assert not victim.exists()


def test_run_bash_timeout(monkeypatch):
    monkeypatch.setattr(tools, "MAX_BASH_SECONDS", 1)
    out = run_bash("sleep 3")
    assert "error" in out and "timeout" in out["error"]


def test_dispatch_unknown_tool():
    out = dispatch("not_a_tool", {})
    assert "error" in out and "unknown" in out["error"]


def test_dispatch_bad_args():
    out = dispatch("read_file", {"wrong_kwarg": "x"})
    assert "error" in out


def test_dispatch_routes_to_tool(tmp_path):
    p = tmp_path / "f.txt"
    p.write_text("hi")
    out = dispatch("read_file", {"path": str(p)})
    assert out["content"] == "hi"


# --- retrieve tool ---------------------------------------------------------

def test_retrieve_tool_returns_results(thandv_home, fake_embed):
    from thandv.rag import ingest_text

    ingest_text("hello world", source="h.md", persona="code")
    out = retrieve("hello", persona="code", k=3)
    assert "error" not in out
    assert out["persona"] == "code"
    assert out["k"] == 3
    assert len(out["results"]) == 1
    assert out["results"][0]["text"] == "hello world"


def test_retrieve_tool_empty_corpus(thandv_home, fake_embed):
    out = retrieve("anything", persona="code", k=5)
    assert out["results"] == []


def test_retrieve_tool_validates_query():
    assert "error" in retrieve("", persona="code")
    assert "error" in retrieve("   ", persona="code")


def test_retrieve_tool_validates_k():
    assert "error" in retrieve("q", persona="code", k=0)
    assert "error" in retrieve("q", persona="code", k=999)
    assert "error" in retrieve("q", persona="code", k="not-an-int")


def test_retrieve_tool_propagates_embed_failure(thandv_home, fake_embed, monkeypatch):
    from thandv import rag
    from thandv.rag import ingest_text

    # Ingest with a working embedder, then break it before the retrieve call
    # so we actually exercise the query-time embed failure path.
    ingest_text("alpha", source="a.md", persona="code")

    def boom(text):
        raise RuntimeError("ollama unreachable")

    monkeypatch.setattr(rag, "embed", boom)
    out = retrieve("anything", persona="code", k=3)
    assert "error" in out
    assert "ollama" in out["error"]


def test_dispatch_retrieve(thandv_home, fake_embed):
    from thandv.rag import ingest_text

    ingest_text("alpha", source="a.md", persona="code")
    out = dispatch("retrieve", {"query": "alpha", "persona": "code", "k": 1})
    assert "error" not in out
    assert out["results"][0]["text"] == "alpha"


# --- Native tool schemas --------------------------------------------------

def test_every_tool_has_a_schema():
    from thandv.tools import TOOL_SCHEMAS, TOOLS

    impl = set(TOOLS)
    sch = {s["function"]["name"] for s in TOOL_SCHEMAS}
    assert impl == sch, f"impl-schema mismatch: only_impl={impl - sch}, only_schema={sch - impl}"


def test_schemas_have_required_fields():
    from thandv.tools import TOOL_SCHEMAS

    for s in TOOL_SCHEMAS:
        assert s["type"] == "function"
        fn = s["function"]
        assert fn["name"] and isinstance(fn["name"], str)
        assert fn["description"] and isinstance(fn["description"], str)
        params = fn["parameters"]
        assert params["type"] == "object"
        assert isinstance(params["properties"], dict)
        # Every required key must exist in properties.
        for req in params.get("required", []):
            assert req in params["properties"], f"{fn['name']}: required {req} missing from properties"


def test_retrieve_schema_includes_persona_override():
    from thandv.tools import TOOL_SCHEMAS

    fn = next(s["function"] for s in TOOL_SCHEMAS if s["function"]["name"] == "retrieve")
    assert "persona" in fn["parameters"]["properties"]
    # persona is NOT in required — the runtime fills it in.
    assert "persona" not in fn["parameters"].get("required", [])
