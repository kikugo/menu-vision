from src.imaging import cache_key, cached_image, store_image


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
