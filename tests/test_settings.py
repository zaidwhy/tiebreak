import pytest

from tiebreak.settings import Settings


def test_defaults_bind_loopback_and_need_no_key():
    cfg = Settings()
    assert cfg.host == "127.0.0.1"
    cfg.check_exposure()


def test_public_bind_without_key_is_refused():
    with pytest.raises(ValueError, match="TIEBREAK_API_KEY"):
        Settings(host="0.0.0.0").check_exposure()


def test_public_bind_with_key_is_allowed():
    Settings(host="0.0.0.0", api_key="k").check_exposure()


def test_env_overrides_and_bad_values_name_the_variable(monkeypatch):
    monkeypatch.setenv("TIEBREAK_MAX_DOCUMENTS", "7")
    assert Settings.from_env().max_documents == 7
    monkeypatch.setenv("TIEBREAK_MAX_DOCUMENTS", "lots")
    with pytest.raises(ValueError, match="TIEBREAK_MAX_DOCUMENTS"):
        Settings.from_env()
    monkeypatch.setenv("TIEBREAK_MAX_DOCUMENTS", "7")
    monkeypatch.setenv("TIEBREAK_SCORES", "percent")
    with pytest.raises(ValueError, match="TIEBREAK_SCORES"):
        Settings.from_env()
