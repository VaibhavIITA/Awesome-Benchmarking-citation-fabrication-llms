import pandas as pd
import pytest

from src.topics import assign_bands


def df(n):
    return pd.DataFrame({"topic": [f"t{i}" for i in range(n)], "works_count": list(range(n, 0, -1))})


def test_bands_and_pilot():
    r = assign_bands(df(90), 20, 2, 4, 42)
    assert len(r) == 66
    assert r.groupby("band").size().to_dict() == {"high": 22, "low": 22, "medium": 22}
    assert r.groupby("band")["is_pilot"].sum().to_dict() == {"high": 2, "low": 2, "medium": 2}
    assert not (r["is_pilot"] & r["is_robust"]).any()
    hi = r[r.band == "high"]["works_count"].min()
    lo = r[r.band == "low"]["works_count"].max()
    assert hi > lo


def test_deterministic():
    assert assign_bands(df(90), 20, 2, 4, 42).equals(assign_bands(df(90), 20, 2, 4, 42))


def test_too_few_topics():
    with pytest.raises(ValueError):
        assign_bands(df(30), 20, 2, 4, 42)


def test_fast_subset_two_per_band():
    import pandas as pd
    from src.generate import subset
    t = pd.DataFrame({"topic_id": [f"T{i:03d}" for i in range(12)],
                      "band": ["high"] * 4 + ["medium"] * 4 + ["low"] * 4,
                      "is_pilot": [True, False, False, False] * 3})
    s = subset(t, 2, 11)
    assert len(s) == 6 and not s["is_pilot"].any()
    assert s.groupby("band").size().tolist() == [2, 2, 2]
    assert s["topic_id"].tolist() == subset(t, 2, 11)["topic_id"].tolist()   # same seed, same topics
