"""install.sh links the venv's `thandv` into the bin dir, replacing an old
link but never a file or directory that something else put there.

Hermetic: `python3` and `ollama` are stubs, so no venv is built, nothing is
downloaded and no model is pulled.
"""

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

pytestmark = pytest.mark.skipif(
    sys.platform == "win32" or shutil.which("bash") is None, reason="install.sh needs bash"
)

# `python3 -m venv DIR` makes a fake venv whose python prints a model name;
# any other call prints a version.
PYTHON3 = """#!/bin/sh
if [ "$1" = "-m" ] && [ "$2" = "venv" ]; then
    mkdir -p "$3/bin"
    printf '#!/bin/sh\\n[ "$1" = "-c" ] && echo stub-model\\nexit 0\\n' > "$3/bin/python"
    printf '#!/bin/sh\\necho thandv-from-venv\\n' > "$3/bin/thandv"
    chmod +x "$3/bin/python" "$3/bin/thandv"
    exit 0
fi
echo 3.12
"""


@pytest.fixture
def install(tmp_path):
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    shutil.copy(REPO / "install.sh", checkout / "install.sh")
    stubs = tmp_path / "stubs"
    stubs.mkdir()
    for name, body in (("python3", PYTHON3), ("ollama", "#!/bin/sh\nexit 0\n")):
        (stubs / name).write_text(body)
        (stubs / name).chmod(0o755)
    bin_dir = tmp_path / "bin"
    env = {
        "HOME": str(tmp_path / "home"),
        "PATH": f"{stubs}:/usr/bin:/bin",
        "THANDV_BIN_DIR": str(bin_dir),
    }

    def run():
        return subprocess.run(
            ["bash", str(checkout / "install.sh")],
            env=env, capture_output=True, text=True, timeout=60,
        )

    return run, checkout / ".venv" / "bin" / "thandv", bin_dir / "thandv"


def test_fresh_install_links_the_venv_command(install):
    run, target, link = install
    r = run()
    assert r.returncode == 0, r.stdout + r.stderr
    assert link.is_symlink() and Path(link.readlink()) == target
    assert "is not on PATH" in r.stdout


def test_an_old_link_is_replaced(install, tmp_path):
    """A re-run, or a moved checkout that left the link dangling."""
    run, target, link = install
    link.parent.mkdir()
    link.symlink_to(tmp_path / "old-checkout" / ".venv" / "bin" / "thandv")
    r = run()
    assert r.returncode == 0, r.stdout + r.stderr
    assert Path(link.readlink()) == target


def test_a_file_at_the_link_path_is_not_clobbered(install):
    run, _, link = install
    link.parent.mkdir()
    link.write_text("#!/bin/sh\necho mine\n")
    r = run()
    assert r.returncode != 0
    assert str(link) in r.stderr and "not a symlink" in r.stderr
    assert not link.is_symlink() and link.read_text() == "#!/bin/sh\necho mine\n"


def test_a_directory_at_the_link_path_is_refused(install):
    """`ln -sf` would put the link inside it and still report success."""
    run, _, link = install
    link.mkdir(parents=True)
    r = run()
    assert r.returncode != 0
    assert list(link.iterdir()) == []


def test_a_link_to_a_directory_is_replaced_not_entered(install, tmp_path):
    run, target, link = install
    other = tmp_path / "other"
    other.mkdir()
    link.parent.mkdir()
    link.symlink_to(other)
    r = run()
    assert r.returncode == 0, r.stdout + r.stderr
    assert Path(link.readlink()) == target
    assert list(other.iterdir()) == []
