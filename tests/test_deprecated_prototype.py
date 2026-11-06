"""prototype in-memory (app/retrieval.py, app/ingest.py) ไม่ผ่าน ACL: import ต้องออก DeprecationWarning และ docstring ต้องบอกชัด"""

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent


def run(module: str, *flags: str) -> subprocess.CompletedProcess:
    code = f"import sys; sys.path.insert(0, {str(ROOT)!r}); import {module}"
    return subprocess.run([sys.executable, *flags, "-c", code], capture_output=True, text=True, encoding="utf-8",
                          env={"SYSTEMROOT": "C:\\Windows", "PYTHONIOENCODING": "utf-8"})


@pytest.mark.parametrize("module", ["app.retrieval", "app.ingest"])
def test_importing_the_prototype_warns_about_acl(module):
    result = run(module, "-W", "always")
    assert result.returncode == 0
    assert "DeprecationWarning" in result.stderr and "ไม่ผ่าน ACL" in result.stderr


@pytest.mark.parametrize("module", ["app.retrieval", "app.ingest"])
def test_warning_can_be_turned_into_an_error(module):
    assert run(module, "-W", "error::DeprecationWarning").returncode != 0


@pytest.mark.parametrize("module", ["retrieval", "ingest"])
def test_docstring_says_deprecated_and_no_acl(module):
    doc = (ROOT / "app" / f"{module}.py").read_text(encoding="utf-8").split('"""')[1]
    assert "DEPRECATED" in doc and "ไม่ผ่าน ACL" in doc
