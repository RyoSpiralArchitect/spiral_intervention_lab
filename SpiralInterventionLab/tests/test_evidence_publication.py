from pathlib import Path
import runpy

import pytest


EXPORT = runpy.run_path(str(Path(__file__).resolve().parents[2] / "scripts/export_intervention_evidence.py"))


def test_public_copy_removes_only_repository_prefix_and_preserves_original(tmp_path):
    original = ('{"' + str(tmp_path) + '/SpiralInterventionLab/code.py":"hash"}').encode()
    assert EXPORT["public_bytes"](original, tmp_path) == b'{"SpiralInterventionLab/code.py":"hash"}'
    assert str(tmp_path).encode() in original


@pytest.mark.parametrize("content", [b"/Users/other/private", b"/home/other/private", b"Bearer " + b"x" * 24])
def test_public_copy_refuses_unreviewed_local_paths_and_secret_like_values(tmp_path, content):
    with pytest.raises(ValueError, match="unreviewed"):
        EXPORT["public_bytes"](content, tmp_path)
