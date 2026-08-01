from src.imaging import cache_key, cached_image, store_image, generate_image


# _call_image_model returns raw bytes now, so tests no longer need to mimic the
# nested Imagen response object.


def _stub_client(monkeypatch):
    """generate_image builds a client before it ever calls Imagen, so a test that
    only stubs _call_imagen dies on the missing API key and never reaches the
    retry loop it means to exercise."""
    monkeypatch.setattr("src.imaging._get_client", lambda: object())


def test_cache_key_is_stable_for_same_inputs():
    assert cache_key("grilled salmon", "rustic") == cache_key("grilled salmon", "rustic")


def test_cache_key_changes_with_style():
    assert cache_key("grilled salmon", "rustic") != cache_key("grilled salmon", "modern")


def test_store_then_read_round_trips(tmp_path, monkeypatch):
    monkeypatch.setattr("src.imaging.CACHE_DIR", tmp_path)
    key = cache_key("tiramisu", "")
    assert cached_image(key) is None
    store_image(key, b"fake-png-bytes")
    assert cached_image(key) == b"fake-png-bytes"


def test_cached_image_returns_none_for_unknown_key(tmp_path, monkeypatch):
    monkeypatch.setattr("src.imaging.CACHE_DIR", tmp_path)
    assert cached_image("nonexistent") is None


def test_generate_image_retries_then_succeeds(monkeypatch, tmp_path):
    monkeypatch.setattr("src.imaging.CACHE_DIR", tmp_path)
    _stub_client(monkeypatch)
    calls = {"n": 0}

    def flaky(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] < 3:
            raise RuntimeError("429 RESOURCE_EXHAUSTED")
        return b"png"

    monkeypatch.setattr("src.imaging._call_image_model", flaky)
    monkeypatch.setattr("src.imaging.RETRY_SLEEP_SECONDS", 0)

    result = generate_image({"name": "soup", "prompt": "a bowl of soup"})

    assert calls["n"] == 3
    assert result["image_bytes"] == b"png"


def test_generate_image_gives_up_after_max_attempts(monkeypatch, tmp_path):
    monkeypatch.setattr("src.imaging.CACHE_DIR", tmp_path)
    _stub_client(monkeypatch)
    attempts = {"n": 0}

    def always_fails(*args, **kwargs):
        attempts["n"] += 1
        raise RuntimeError("429 RESOURCE_EXHAUSTED")

    monkeypatch.setattr("src.imaging._call_image_model", always_fails)
    monkeypatch.setattr("src.imaging.RETRY_SLEEP_SECONDS", 0)

    assert generate_image({"name": "soup", "prompt": "a bowl of soup"}) is None
    # without this the test also passes when the client blows up and Imagen is
    # never called at all, which is not what it is meant to prove
    assert attempts["n"] == 3
