"""pytest suite for load_week3.py -- no network, no API keys, in-memory SQLite."""
from datetime import date

import numpy as np
import pytest

import load_week3 as lw
from sample_data import sample_locations, sample_providers

SNAP = date(2026, 9, 17)


@pytest.fixture
def conn():
    c = lw.connect(":memory:")
    yield c
    c.close()


def fact_count(conn, snap=None):
    if snap is None:
        return conn.execute("SELECT COUNT(*) FROM fact_location_snapshot").fetchone()[0]
    return conn.execute("SELECT COUNT(*) FROM fact_location_snapshot WHERE snapshot_date=?",
                        (snap.isoformat(),)).fetchone()[0]


# --- helpers ---------------------------------------------------------------
def test_is_blank_treats_nan_as_blank():
    assert lw.is_blank(np.nan) and lw.is_blank(None) and lw.is_blank("  ")
    assert not lw.is_blank("Good")


def test_rating_period_key_month_grain_and_not_rated():
    assert lw.rating_period_key("Good", "2024-05-14") == "2024-05"
    assert lw.rating_period_key("Good", "2024-05-14T00:00:00") == "2024-05"
    assert lw.rating_period_key(np.nan, np.nan) == lw.NOT_RATED_KEY


# --- happy path & idempotency ---------------------------------------------
def test_clean_sample_loads(conn):
    status, report = lw.run_pipeline(conn, sample_locations(), sample_providers(), SNAP)
    assert status == "LOADED", report.to_frame()
    assert fact_count(conn) == len(sample_locations())


def test_rerun_same_snapshot_is_idempotent(conn):
    for _ in range(3):
        lw.run_pipeline(conn, sample_locations(), sample_providers(), SNAP)
    assert fact_count(conn) == len(sample_locations())
    assert conn.execute("SELECT COUNT(*) FROM dim_provider").fetchone()[0] == len(sample_providers())
    assert conn.execute("SELECT COUNT(*) FROM dim_area").fetchone()[0] == 7


def test_new_snapshot_date_adds_history_not_duplicates(conn):
    lw.run_pipeline(conn, sample_locations(), sample_providers(), SNAP)
    lw.run_pipeline(conn, sample_locations(), sample_providers(), date(2026, 10, 17))
    assert fact_count(conn) == 2 * len(sample_locations())


def test_rerun_with_corrected_rating_replaces_not_appends(conn):
    lw.run_pipeline(conn, sample_locations(), sample_providers(), SNAP)
    locs = sample_locations()
    locs.loc[locs["location_id"] == "1-100003", "overall_rating"] = "Good"
    lw.run_pipeline(conn, locs, sample_providers(), SNAP)
    rows = conn.execute("SELECT overall_rating FROM fact_location_snapshot "
                        "WHERE location_id='1-100003'").fetchall()
    assert rows == [("Good",)]


def test_provider_dimension_upserts_changed_name(conn):
    lw.run_pipeline(conn, sample_locations(), sample_providers(), SNAP)
    provs = sample_providers()
    provs.loc[0, "provider_name"] = "Home Safe Care Group Ltd"
    lw.run_pipeline(conn, sample_locations(), provs, SNAP)
    names = conn.execute("SELECT provider_name FROM dim_provider WHERE provider_id='1-000001'").fetchall()
    assert names == [("Home Safe Care Group Ltd",)]


def test_unrated_location_gets_not_rated_key(conn):
    lw.run_pipeline(conn, sample_locations(), sample_providers(), SNAP)
    row = conn.execute("SELECT rating_period_key, is_rated, overall_rating FROM fact_location_snapshot "
                       "WHERE location_id='1-100006'").fetchone()
    assert row == (lw.NOT_RATED_KEY, 0, None)


# --- gates block and leave the database unchanged --------------------------
def test_duplicate_location_blocks_and_writes_nothing(conn):
    lw.run_pipeline(conn, sample_locations(), sample_providers(), SNAP)
    bad = sample_locations()
    bad.loc[1, "location_id"] = bad.loc[0, "location_id"]
    status, report = lw.run_pipeline(conn, bad, sample_providers(), SNAP)
    assert status == "BLOCKED"
    assert "location_id_unique" in [r.name for r in report.blocking_failures]
    assert fact_count(conn) == len(sample_locations())  # previous good load untouched


def test_duplicate_provider_id_blocks(conn):
    provs = sample_providers()
    provs.loc[1, "provider_id"] = provs.loc[0, "provider_id"]
    status, report = lw.run_pipeline(conn, sample_locations(), provs, SNAP)
    assert status == "BLOCKED" and fact_count(conn) == 0


def test_orphan_provider_blocks(conn):
    locs = sample_locations()
    locs.loc[0, "provider_id"] = "1-999999"
    status, report = lw.run_pipeline(conn, locs, sample_providers(), SNAP)
    assert status == "BLOCKED"
    assert "every_location_has_a_provider" in [r.name for r in report.blocking_failures]


def test_unknown_rating_label_blocks(conn):
    locs = sample_locations()
    locs.loc[0, "overall_rating"] = "Excellent"
    status, _ = lw.run_pipeline(conn, locs, sample_providers(), SNAP)
    assert status == "BLOCKED"


def test_future_rating_date_blocks(conn):
    locs = sample_locations()
    locs.loc[0, "overall_rating_date"] = "2027-01-01"
    status, _ = lw.run_pipeline(conn, locs, sample_providers(), SNAP)
    assert status == "BLOCKED"


def test_missing_column_blocks_cleanly(conn):
    status, report = lw.run_pipeline(conn, sample_locations().drop(columns=["postal_code"]),
                                     sample_providers(), SNAP)
    assert status == "BLOCKED"
    assert report.blocking_failures[0].name == "required_columns_present"


def test_one_bad_postcode_in_many_is_a_warning_not_a_block(conn):
    locs = sample_locations()
    big = locs.loc[locs.index.repeat(3)].reset_index(drop=True)   # 24 rows
    big["location_id"] = [f"1-2{i:05d}" for i in range(len(big))]
    big.loc[0, "postal_code"] = "NOT A POSTCODE"                   # 1/24 = 4.2%
    status, report = lw.run_pipeline(conn, big, sample_providers(), SNAP)
    assert status == "LOADED"
    assert "all_postcodes_valid" in [r.name for r in report.warnings]
    assert conn.execute("SELECT postcode_valid FROM fact_location_snapshot "
                        "WHERE postal_code='NOT A POSTCODE'").fetchone() == (0,)


def test_many_bad_postcodes_block(conn):
    locs = sample_locations()
    locs.loc[:1, "postal_code"] = "XXX"   # 2/8 = 25%
    status, _ = lw.run_pipeline(conn, locs, sample_providers(), SNAP)
    assert status == "BLOCKED"


def test_non_kent_authority_is_flagged_not_dropped(conn):
    locs = sample_locations()
    locs.loc[0, "local_authority"] = "Medway"
    status, report = lw.run_pipeline(conn, locs, sample_providers(), SNAP)
    assert status == "LOADED"
    assert "local_authority_in_kent_list" in [r.name for r in report.warnings]
    assert conn.execute("SELECT is_kent_district FROM dim_area WHERE local_authority='Medway'").fetchone() == (0,)


def test_row_count_drift_blocks_then_override_allows(conn):
    lw.run_pipeline(conn, sample_locations(), sample_providers(), SNAP)
    half = sample_locations().head(4)                       # 50% drop vs last snapshot
    nxt = date(2026, 10, 17)
    status, _ = lw.run_pipeline(conn, half, sample_providers(), nxt)
    assert status == "BLOCKED" and fact_count(conn, nxt) == 0
    status, _ = lw.run_pipeline(conn, half, sample_providers(), nxt, allow_row_count_change=True)
    assert status == "LOADED" and fact_count(conn, nxt) == 4


def test_failure_mid_load_rolls_back(conn, monkeypatch):
    lw.run_pipeline(conn, sample_locations(), sample_providers(), SNAP)

    def boom(*a, **k):
        raise RuntimeError("simulated crash after DELETE")
    real_upsert = lw._upsert_rating_periods

    def crash_after_delete(c, keys):
        real_upsert(c, keys)
        c.execute("DELETE FROM fact_location_snapshot WHERE snapshot_date=?", (SNAP.isoformat(),))
        boom()
    monkeypatch.setattr(lw, "_upsert_rating_periods", crash_after_delete)
    with pytest.raises(RuntimeError):
        lw.run_pipeline(conn, sample_locations(), sample_providers(), SNAP)
    assert fact_count(conn) == len(sample_locations())   # delete was rolled back
    assert conn.execute("SELECT status FROM load_audit ORDER BY rowid DESC LIMIT 1").fetchone() == ("FAILED",)


def test_every_run_is_audited(conn):
    lw.run_pipeline(conn, sample_locations(), sample_providers(), SNAP)
    bad = sample_locations()
    bad.loc[0, "overall_rating"] = "Excellent"
    lw.run_pipeline(conn, bad, sample_providers(), SNAP)
    statuses = [r[0] for r in conn.execute("SELECT status FROM load_audit ORDER BY rowid")]
    assert statuses == ["LOADED", "BLOCKED"]


def test_density_query_runs_and_excludes_deregistered(conn):
    lw.run_pipeline(conn, sample_locations(), sample_providers(), SNAP)
    rows = {r[0]: r for r in conn.execute(lw.DENSITY_BY_AREA_SQL)}
    assert "Dover" not in rows                   # only location there is Deregistered
    assert rows["Canterbury"][2] == 2            # two registered locations
    # Dartford's only location is unrated: counts must be 0, not NULL (NULL-comparison trap)
    assert rows["Dartford"][3:8] == (0, 0, 0, 0, 1)
    assert rows["Dartford"][8] is None           # no rated locations -> share undefined, not 0
