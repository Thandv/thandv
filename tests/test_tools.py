
from thandv import tools
from thandv.tools import dispatch, edit_file, list_dir, read_file, run_bash, write_file


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
