"""README.md and STATUS.md must not name commands, files or modules that do not exist."""

import importlib.util
import re
from pathlib import Path

import pytest

from thandv import cli

REPO = Path(__file__).resolve().parents[1]
DOCS = ["README.md", "STATUS.md"]

LINK_RE = re.compile(r"\]\(([^)\s]+)\)")
INLINE_RE = re.compile(r"`([^`\n]+)`")
FENCE_RE = re.compile(r"```[^\n]*\n(.*?)```", re.DOTALL)
# `thandv <cmd> [<action>]`, but not doctor output such as `thandv version: 0.10.0`.
COMMAND_RE = re.compile(r"(?:^|[\s`(])thandv ([a-z][\w-]*(?: [a-z][\w-]*)?)(?![\w:-])")
# Dotted module names, but not a path or URL ending in `thandv.git`.
MODULE_RE = re.compile(r"(?<![\w/.:-])thandv(?:\.[a-z_]+)+\b")
FILE_EXT = (".md", ".py", ".sh", ".toml", ".yml", ".json", ".txt")


def _text(name: str) -> str:
    return (REPO / name).read_text()


def _choices(argv: list[str], capsys) -> set[str]:
    """Subcommand names listed as `{a,b,c}` in the real parser's --help."""
    with pytest.raises(SystemExit):
        cli.main([*argv, "--help"])
    found = re.search(r"\{([\w,-]+)\}", capsys.readouterr().out)
    return set(found.group(1).split(",")) if found else set()


def _subcommands(capsys) -> dict[str, set[str]]:
    """Top-level subcommands, each with its second-level actions (if any)."""
    return {name: _choices([name], capsys) for name in _choices([], capsys)}


def _path_like(token: str) -> bool:
    if token.startswith(("~", "/", "$", "-", "http")) or any(c in token for c in " <>*{}"):
        return False
    return "/" in token or token.endswith(FILE_EXT)


@pytest.mark.parametrize("doc", DOCS)
def test_doc_exists(doc):
    assert (REPO / doc).is_file()


@pytest.mark.parametrize("doc", DOCS)
def test_linked_and_named_files_exist(doc):
    text = _text(doc)
    missing = []
    for target in LINK_RE.findall(text):
        target = target.split("#", 1)[0]
        if not target or target.startswith(("http://", "https://", "mailto:")):
            continue
        if not (REPO / doc).parent.joinpath(target).exists():
            missing.append(target)
    for token in INLINE_RE.findall(text):
        if _path_like(token) and not (REPO / token).exists():
            missing.append(token)
    assert missing == []


@pytest.mark.parametrize("doc", DOCS)
def test_named_commands_exist(doc, capsys):
    commands = _subcommands(capsys)
    text = _text(doc)
    code = "\n".join(FENCE_RE.findall(text)) + "\n" + "\n".join(INLINE_RE.findall(text))
    unknown = []
    for match in COMMAND_RE.finditer(code):
        words = match.group(1).split()
        if words[0] not in commands:
            unknown.append(match.group(1))
        elif len(words) > 1 and commands[words[0]] and words[1] not in commands[words[0]]:
            unknown.append(match.group(1))
    assert unknown == []


@pytest.mark.parametrize("doc", DOCS)
def test_named_modules_exist(doc):
    missing = [m for m in set(MODULE_RE.findall(_text(doc))) if importlib.util.find_spec(m) is None]
    assert missing == []


def test_readme_layout_matches_the_tree():
    """Every entry in README's Layout block exists: unindented entries at the
    repo root, indented ones inside the `thandv/` package."""
    text = _text("README.md")
    block = FENCE_RE.findall(text[text.index("## Layout"):])[0]
    missing = []
    for line in block.splitlines():
        if not line.strip():
            continue
        entry = line.split()[0]
        path = REPO / entry if not line.startswith(" ") else REPO / "thandv" / entry
        if not path.exists():
            missing.append(line.strip())
    assert missing == []


def test_development_doc_suite_time_is_not_the_old_claim():
    assert "~1.2s" not in _text("docs/DEVELOPMENT.md")
