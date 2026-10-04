"""`.env` parsing and API-key handling."""
import os

import pytest

from touchscale_qc import config as C


def test_dotenv_parsing(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("# comment\n"
                   "QC_T_PLAIN=abc\n"
                   "export QC_T_EXPORT=def\n"
                   "QC_T_INLINE=ghi   # trailing comment\n"
                   "QC_T_QUOTED='has # hash'\n"
                   "QC_T_SET=from_file\n")
    for k in ["QC_T_PLAIN", "QC_T_EXPORT", "QC_T_INLINE", "QC_T_QUOTED"]:
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("QC_T_SET", "from_env")
    C._load_dotenv(str(env))
    assert os.environ["QC_T_PLAIN"] == "abc"
    assert os.environ["QC_T_EXPORT"] == "def"
    assert os.environ["QC_T_INLINE"] == "ghi"
    assert os.environ["QC_T_QUOTED"] == "has # hash"
    assert os.environ["QC_T_SET"] == "from_env"              # environment wins


def test_missing_api_key_is_actionable(monkeypatch):
    monkeypatch.setattr(C, "API_KEY", "")
    with pytest.raises(RuntimeError, match="QC_API_KEY"):
        C.require_api_key()
