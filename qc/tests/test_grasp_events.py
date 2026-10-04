"""Step-1 responses must be normalised before numpy touches them."""
import numpy as np
import pytest

from touchscale_qc import grasp_events as Q


def test_dict_shaped_free_interval_is_coerced():
    # the model sometimes answers [{"start", "end"}] instead of [[start, end]]
    bad = {"free": [{"start": 15.7, "end": 18.0}],
           "contacts": [{"start": 0.0, "end": 1.7, "object": "wooden block"}]}
    t = np.linspace(0, 20, 200)
    with pytest.raises(TypeError):         # the raw shape really does break downstream code
        for a, b in bad["free"]:
            _ = (t >= a) & (t <= b)
    r = Q.normalize_r1(dict(bad))
    assert r["free"] == [[15.7, 18.0]]
    for a, b in r["free"]:
        _ = (t >= a) & (t <= b)


def test_documented_shape_passes_through():
    good = {"free": [[1.0, 2.0], [3.0, 4.0]], "contacts": []}
    assert Q.normalize_r1(dict(good))["free"] == [[1.0, 2.0], [3.0, 4.0]]


def test_numeric_strings_are_salvaged_and_junk_dropped():
    assert Q.normalize_r1({"free": [["1.5", "2.5"]]})["free"] == [[1.5, 2.5]]
    assert Q.normalize_r1({"free": [None, "x", [1.0]]})["free"] == []
    assert Q.normalize_r1({"contacts": [{"object": "x"}]})["contacts"] == []
    assert Q.normalize_r1(None) == {}


def test_parse_json_tolerates_fences_and_prose():
    assert Q.parse_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert Q.parse_json('Sure! {"a": 2} hope this helps') == {"a": 2}
    assert Q.parse_json("no json here") == {}


def test_segments_bridges_short_gaps():
    t = np.arange(20) / 10.0
    m = np.zeros(20, bool)
    m[2:6] = True
    m[7:9] = True                          # 1-sample gap -> bridged
    m[15:18] = True                        # separate run
    assert Q.segments(m, t) == [(0.2, 0.8), (1.5, 1.7)]
    assert Q.segments(m, t, min_len=0.5) == [(0.2, 0.8)]
