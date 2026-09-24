"""pytest suite for homesafe_market.py — no network, no API keys."""
import numpy as np
import pandas as pd
import pytest

import homesafe_market as hm


# --- cleaning ---------------------------------------------------------------
def test_normalise_postcode_spacing_and_case():
    assert hm.normalise_postcode("tn236ln") == "TN23 6LN"
    assert hm.normalise_postcode("TN23  6LN") == "TN23 6LN"
    assert hm.normalise_postcode(np.nan) is None


def test_outward_code_handles_short_and_long_districts():
    assert hm.outward_code("TN23 6LN") == "TN23"
    assert hm.outward_code("CT1 3AA") == "CT1"          # 3-char outcode
    assert hm.outward_code("ME10 1NR") == "ME10"
    assert hm.outward_code(None) is None


def test_outward_code_does_not_mangle_short_postcodes():
    # 'CT2 8AJ' written without a space is 'CT28AJ' — a naive regex reads this
    # as outcode 'CT28' (which does not exist). Stripping the last three
    # characters is the only safe rule.
    assert hm.outward_code("CT2 8AJ") == "CT2"
    assert hm.outward_code("CT28AJ") == "CT2"


def test_postcode_validity():
    assert hm.is_valid_uk_postcode("TN23 6LN")
    assert not hm.is_valid_uk_postcode("NOT A POSTCODE")
    assert not hm.is_valid_uk_postcode(np.nan)


def test_is_blank_treats_nan_as_blank():
    assert hm.is_blank(np.nan) and hm.is_blank("   ") and not hm.is_blank("x")


def test_has_service_is_exact_not_substring():
    df = pd.DataFrame({"service_types": ["Homecare agencies|Supported living",
                                         "Homecare agencies (specialist)",
                                         "Doctors/GPs"]})
    assert hm.has_service(df, hm.HOMECARE_SERVICE).tolist() == [True, False, False]


# --- geometry ---------------------------------------------------------------
def test_haversine_against_known_distance():
    # Ashford (TN23 6LN) to Canterbury city centre (CT1 2AA): ~13-14 miles
    d = hm.haversine_miles(51.1323, 0.867371, 51.282506, 1.081612)
    assert 12.5 < d < 15.0


def test_haversine_zero_for_same_point():
    assert hm.haversine_miles(51.1323, 0.867371, 51.1323, 0.867371) == pytest.approx(0, abs=1e-9)


def test_rings_are_assigned_by_distance():
    df = pd.DataFrame({"lat": [51.1323, 51.282506], "lon": [0.867371, 1.081612]})
    out = hm.add_distance(df, 51.1323, 0.867371)
    assert str(out.loc[0, "ring"]) == "0-5 mi"
    assert str(out.loc[1, "ring"]) == "10-15 mi"


# --- market metrics ---------------------------------------------------------
def test_hhi_monopoly_and_fragmented():
    assert hm.herfindahl_index([10]) == pytest.approx(10000)
    assert hm.herfindahl_index([1] * 100) == pytest.approx(100)
    assert hm.herfindahl_index([]) == 0.0


# --- geocoding --------------------------------------------------------------
def test_geocode_uses_cache_and_never_calls_network_when_disabled():
    cache = pd.DataFrame({"pc": ["TN236LN"], "lat": [51.1323], "lon": [0.867371],
                          "district": ["Ashford"]})
    out = hm.geocode_postcodes(["TN236LN", "CT12AA"], cache=cache, allow_network=False)
    assert list(out["pc"]) == ["TN236LN"]      # the miss is reported, not fetched


def test_geocode_fetches_only_cache_misses():
    class FakeResponse:
        def __init__(self, payload): self._p = payload
        def raise_for_status(self): pass
        def json(self): return self._p

    class FakeSession:
        def __init__(self): self.calls = []
        def post(self, url, json, timeout):
            self.calls.append(json["postcodes"])
            return FakeResponse({"result": [{"query": pc, "result": {
                "latitude": 51.0, "longitude": 1.0, "admin_district": "Testshire"}}
                for pc in json["postcodes"]]})

    cache = pd.DataFrame({"pc": ["TN236LN"], "lat": [51.1323], "lon": [0.867371],
                          "district": ["Ashford"]})
    session = FakeSession()
    out = hm.geocode_postcodes(["TN236LN", "CT12AA"], cache=cache, allow_network=True,
                               session=session)
    assert session.calls == [["CT12AA"]]        # cached postcode not re-requested
    assert set(out["pc"]) == {"TN236LN", "CT12AA"}


# --- pipeline ---------------------------------------------------------------
def sample_locations():
    return pd.DataFrame([
        {"location_id": "1-1", "provider_id": "1-p1", "location_name": "Alpha Care",
         "postcode": "TN23 6LN", "outcode": "TN23", "district": "Ashford",
         "lat": 51.1323, "lon": 0.867371, "miles_from_anchor": 0.0, "in_catchment": True,
         "website": "https://a.example", "latest_check": pd.Timestamp("2025-03-01"),
         "specialisms": "Dementia", "service_types": "Homecare agencies",
         "provider_name": "Alpha Ltd"},
        {"location_id": "1-2", "provider_id": "1-p2", "location_name": "Beta Care",
         "postcode": "CT1 2AA", "outcode": "CT1", "district": "Canterbury",
         "lat": 51.282506, "lon": 1.081612, "miles_from_anchor": 13.6, "in_catchment": True,
         "website": np.nan, "latest_check": pd.NaT, "specialisms": "Dementia",
         "service_types": "Homecare agencies|Supported living", "provider_name": "Beta Ltd"},
    ])


@pytest.fixture
def conn():
    c = hm.connect(":memory:")
    yield c
    c.close()


def fact_rows(conn):
    return conn.execute("SELECT COUNT(*) FROM fact_homecare_location").fetchone()[0]


def test_clean_load_passes_every_gate(conn):
    status, report = hm.run_pipeline(conn, sample_locations(), "2026-09-16")
    assert status == "LOADED", report.to_frame()
    assert fact_rows(conn) == 2


def test_load_is_idempotent(conn):
    for _ in range(3):
        hm.run_pipeline(conn, sample_locations(), "2026-09-16")
    assert fact_rows(conn) == 2
    assert conn.execute("SELECT COUNT(*) FROM dim_provider").fetchone()[0] == 2


def test_new_snapshot_keeps_history(conn):
    hm.run_pipeline(conn, sample_locations(), "2026-09-16")
    hm.run_pipeline(conn, sample_locations(), "2026-10-16")
    assert fact_rows(conn) == 4


def test_missing_website_recorded_as_zero_not_null(conn):
    hm.run_pipeline(conn, sample_locations(), "2026-09-16")
    got = dict(conn.execute("SELECT location_id, has_website FROM fact_homecare_location"))
    assert got == {"1-1": 1, "1-2": 0}


def test_duplicate_location_id_blocks_and_writes_nothing(conn):
    bad = pd.concat([sample_locations(), sample_locations().head(1)], ignore_index=True)
    status, report = hm.run_pipeline(conn, bad, "2026-09-16")
    assert status == "BLOCKED"
    assert "location_id_unique" in [r["check"] for r in report.blocking_failures]
    assert fact_rows(conn) == 0


def test_non_homecare_row_blocks(conn):
    bad = sample_locations()
    bad.loc[1, "service_types"] = "Doctors/GPs"
    status, _ = hm.run_pipeline(conn, bad, "2026-09-16")
    assert status == "BLOCKED"


def test_low_geocoding_coverage_blocks(conn):
    bad = sample_locations()
    bad.loc[1, "lat"] = np.nan
    status, report = hm.run_pipeline(conn, bad, "2026-09-16")   # 50% geocoded
    assert status == "BLOCKED"
    assert "geocoding_coverage" in [r["check"] for r in report.blocking_failures]


def test_coordinates_outside_gb_block(conn):
    bad = sample_locations()
    bad.loc[1, "lat"] = 40.7      # New York
    bad.loc[1, "lon"] = -74.0
    status, _ = hm.run_pipeline(conn, bad, "2026-09-16")
    assert status == "BLOCKED"


def test_one_missing_geocode_in_many_is_a_warning(conn):
    many = pd.concat([sample_locations()] * 20, ignore_index=True)
    many["location_id"] = [f"1-{i}" for i in range(len(many))]
    many.loc[0, "lat"] = np.nan                     # 1/40 missing = 97.5% coverage
    status, report = hm.run_pipeline(conn, many, "2026-09-16")
    assert status == "LOADED"
    assert "all_rows_geocoded" in [r["check"] for r in report.warnings]


def test_audit_records_blocked_and_loaded_runs(conn):
    hm.run_pipeline(conn, sample_locations(), "2026-09-16")
    bad = sample_locations()
    bad.loc[1, "service_types"] = "Dentist"
    hm.run_pipeline(conn, bad, "2026-09-16")
    statuses = [r[0] for r in conn.execute("SELECT status FROM load_audit ORDER BY rowid")]
    assert statuses == ["LOADED", "BLOCKED"]


def test_opportunity_table_counts_supply_and_demand_proxies():
    catchment = sample_locations()
    all_locs = pd.DataFrame({
        "outcode": ["TN23", "TN23", "CT1", "TN29"],
        "service_types": ["Doctors/GPs", "Residential homes", "Nursing homes", "Residential homes"],
    })
    geo = pd.DataFrame({"outcode": ["TN23", "CT1", "TN29"],
                        "lat": [51.1395, 51.2774, 50.9942],
                        "lon": [0.8558, 1.0856, 0.9323],
                        "district": ["Ashford", "Canterbury", "Folkestone and Hythe"]})
    t = hm.opportunity_table(catchment, all_locs, geo, 51.1323, 0.867371)
    assert t.loc["TN23", "homecare"] == 1 and t.loc["TN23", "gp"] == 1
    assert t.loc["TN23", "care_homes"] == 1
    assert t.loc["TN29", "homecare"] == 0          # no supply, demand proxy present
    assert t.loc["TN29", "care_homes"] == 1
    assert pd.isna(t.loc["CT1", "homecare_per_care_home"]) is False
