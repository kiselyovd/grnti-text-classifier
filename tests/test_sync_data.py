"""Smoke-test sync_data idempotence when target files already present."""
from __future__ import annotations
import os, subprocess, sys
from pathlib import Path
import pytest


@pytest.mark.skipif(sys.platform == "win32", reason="bash path-interop on Windows")
def test_sync_data_idempotent_when_files_present(tmp_path: Path) -> None:
    work = tmp_path / "repo"
    (work / "data" / "raw").mkdir(parents=True)
    for name in ("train.jsonl", "test.jsonl"):
        (work / "data" / "raw" / name).write_text('{"dummy":1}\n', encoding="utf-8")
    script = Path(__file__).resolve().parents[1] / "scripts" / "sync_data.sh"
    env = os.environ.copy()
    env["GRNTI_REPO_ROOT"] = str(work)
    r = subprocess.run(["bash", str(script)], env=env, capture_output=True, text=True, cwd=work)
    assert r.returncode == 0, r.stderr
