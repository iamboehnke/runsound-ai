def test_cached_run_exists():
    import json
    from pathlib import Path
    p = Path("data/latest_run.json")
    assert p.exists()
    j = json.loads(p.read_text())
    assert "type" in j
    assert j["type"] == "Run"
    assert "id" in j