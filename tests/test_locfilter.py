"""F6: the location filter must not discard nearly every job by default."""
from applypilot.locfilter import load_location_filter, location_ok


def test_empty_accept_keeps_non_remote():
    assert location_ok("San Francisco, CA", [], []) is True


def test_reject_pattern_still_blocks_with_empty_accept():
    assert location_ok("Pune, India", [], ["india"]) is False


def test_explicit_accept_list_is_exclusive():
    assert location_ok("Oakland", ["Bay Area"], []) is False
    assert location_ok("Bay Area office", ["Bay Area"], []) is True


def test_remote_always_passes():
    assert location_ok("Remote - US", ["Bay Area"], ["india"]) is True


def test_none_location_passes():
    assert location_ok(None, ["Bay Area"], []) is True


def test_loads_current_schema():
    cfg = {"location": {"accept_patterns": ["San Francisco", "Remote"],
                        "reject_patterns": ["India"]}}
    accept, reject = load_location_filter(cfg)
    assert accept == ["San Francisco", "Remote"]
    assert reject == ["India"]


def test_loads_legacy_schema():
    cfg = {"location_accept": ["NYC"], "location_reject_non_remote": ["China"]}
    accept, reject = load_location_filter(cfg)
    assert accept == ["NYC"]
    assert reject == ["China"]


def test_empty_config_yields_empty_lists():
    accept, reject = load_location_filter({})
    assert accept == [] and reject == []
