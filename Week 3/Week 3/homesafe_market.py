"""Home Safe — Ashford (TN23 6LN) domiciliary care market analysis.

Reusable, testable functions for the Week 1–4 walkthrough notebook:
  * cleaning and postcode handling (Week 2 rules)
  * geocoding with an on-disk cache and a postcodes.io fallback (network only
    when the cache misses, so the notebook runs offline)
  * haversine distance and drive-time-proxy catchment rings
  * data-quality gates + an idempotent SQLite star-schema load (Week 3 pattern)
  * market metrics: density, concentration (HHI) and an opportunity score

No network calls happen at import time. Every function below is unit-tested in
test_homesafe_market.py.
"""
from __future__ import annotations

import math
import re
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np
import pandas as pd

# --- constants --------------------------------------------------------------
ANCHOR_POSTCODE = "TN23 6LN"          # Home Safe's assumed base, Ashford
EARTH_RADIUS_MILES = 3958.8

# Straight-line proxy for a 30-minute drive. See the notebook for why this is a
# proxy and not a fact: real drive time follows the M20 and A2 much further than
# it follows a country lane across Romney Marsh.
CATCHMENT_MILES = 20.0
RING_EDGES = [0, 5, 10, 15, 20]

UK_POSTCODE_RE = re.compile(r"^[A-Z]{1,2}\d[A-Z\d]?\s?\d[A-Z]{2}$")
HOMECARE_SERVICE = "Homecare agencies"
GP_SERVICE = "Doctors/GPs"
CARE_HOME_SERVICES = ("Residential homes", "Nursing homes")

OGL_ATTRIBUTION = ("Contains Care Quality Commission information licensed under "
                   "the Open Government Licence v3.0.")


# --- Week 2 cleaning helpers (same rules as the Week 2/3 material) -----------
def is_blank(value) -> bool:
    """NaN-aware blank test. NaN is truthy in Python, so never use `if value`
    on a pandas cell."""
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return str(value).strip() == "" or str(value).strip().lower() in {"nan", "nat", "none"}


def normalise_postcode(raw):
    """Upper-case, single space before the inward code. Returns None if blank."""
    if is_blank(raw):
        return None
    p = re.sub(r"\s+", "", str(raw)).upper()
    if len(p) < 5:
        return p
    return f"{p[:-3]} {p[-3:]}"


def postcode_key(raw):
    """Space-free key used for joining to the geocode cache."""
    pc = normalise_postcode(raw)
    return None if pc is None else pc.replace(" ", "")


def is_valid_uk_postcode(raw) -> bool:
    pc = normalise_postcode(raw)
    return bool(pc) and bool(UK_POSTCODE_RE.match(pc))


def outward_code(raw):
    """'TN23 6LN' -> 'TN23'. The postcode district, used as the analysis area."""
    pc = postcode_key(raw)
    if not pc or len(pc) < 5:
        return None
    return pc[:-3]


# --- extract / load of the CQC bulk directory -------------------------------
def read_cqc_directory(path: str) -> pd.DataFrame:
    """The published Care Directory CSV carries four title rows before the real
    header, so the header row is row 5 (skiprows=4). Everything is read as text:
    postcodes and CQC IDs are identifiers, not numbers."""
    df = pd.read_csv(path, skiprows=4, dtype=str)
    df.columns = [c.strip() for c in df.columns]
    return df


COLUMN_RENAMES = {
    "Name": "location_name",
    "Also known as": "also_known_as",
    "Address": "address",
    "Postcode": "postcode_raw",
    "Phone number": "phone",
    "Service's website (if available)": "website",
    "Service types": "service_types",
    "Date of latest check": "latest_check_raw",
    "Specialisms/services": "specialisms",
    "Provider name": "provider_name",
    "Local authority": "local_authority",
    "Region": "region",
    "Location URL": "location_url",
    "CQC Location ID (for office use only)": "location_id",
    "CQC Provider ID (for office use only)": "provider_id",
}


def tidy_directory(df: pd.DataFrame) -> pd.DataFrame:
    """Rename to snake_case, normalise postcodes, derive outward code and parse
    the inspection date. Nothing is dropped here — filtering comes later."""
    out = df.rename(columns=COLUMN_RENAMES).copy()
    out["postcode"] = out["postcode_raw"].apply(normalise_postcode)
    out["pc_key"] = out["postcode_raw"].apply(postcode_key)
    out["postcode_valid"] = out["postcode_raw"].apply(is_valid_uk_postcode)
    out["outcode"] = out["postcode_raw"].apply(outward_code)
    out["latest_check"] = pd.to_datetime(
        out["latest_check_raw"].str.split(" - ").str[0], format="%d/%b/%Y", errors="coerce")
    for col in ("service_types", "specialisms"):
        out[col] = out[col].fillna("")
    return out


def has_service(df: pd.DataFrame, *service_names) -> pd.Series:
    """Service types are pipe-separated in one cell, so an exact match needs the
    split — a plain `.str.contains('Homecare agencies')` would also hit a future
    value such as 'Homecare agencies (specialist)'."""
    sets = df["service_types"].fillna("").apply(lambda s: {p.strip() for p in s.split("|")})
    wanted = set(service_names)
    return sets.apply(lambda s: bool(s & wanted))


# --- geocoding ---------------------------------------------------------------
def load_geo_cache(path: str) -> pd.DataFrame:
    cache = pd.read_csv(path, dtype={"pc": str})
    cache["lat"] = pd.to_numeric(cache["lat"], errors="coerce")
    cache["lon"] = pd.to_numeric(cache["lon"], errors="coerce")
    return cache


def geocode_postcodes(pc_keys, cache: pd.DataFrame | None = None, allow_network: bool = False,
                      session=None) -> pd.DataFrame:
    """Return a DataFrame [pc, lat, lon, district] for the given space-free
    postcodes. Cache first; only the misses go to postcodes.io (free, no key,
    100 postcodes per POST) and only when allow_network=True."""
    wanted = sorted({k for k in pc_keys if k})
    have = cache[cache["pc"].isin(wanted)] if cache is not None else pd.DataFrame(
        columns=["pc", "lat", "lon", "district"])
    missing = [k for k in wanted if k not in set(have["pc"])]
    if not missing or not allow_network:
        if missing:
            print(f"{len(missing)} postcode(s) not in cache and network lookup is off; "
                  f"they will have no coordinates.")
        return have.reset_index(drop=True)

    import requests
    session = session or requests.Session()
    rows = []
    for i in range(0, len(missing), 100):
        batch = missing[i:i + 100]
        resp = session.post("https://api.postcodes.io/postcodes",
                            json={"postcodes": batch}, timeout=30)
        resp.raise_for_status()
        for item in resp.json()["result"]:
            res = item.get("result")
            rows.append({"pc": item["query"].replace(" ", "").upper(),
                         "lat": res["latitude"] if res else np.nan,
                         "lon": res["longitude"] if res else np.nan,
                         "district": (res or {}).get("admin_district")})
    return pd.concat([have, pd.DataFrame(rows)], ignore_index=True)


# --- geometry ----------------------------------------------------------------
def haversine_miles(lat1, lon1, lat2, lon2):
    """Great-circle distance in miles. Works on scalars or numpy arrays."""
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    dlat, dlon = lat2 - lat1, lon2 - lon1
    h = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_MILES * np.arcsin(np.sqrt(h))


def add_distance(df: pd.DataFrame, anchor_lat: float, anchor_lon: float) -> pd.DataFrame:
    out = df.copy()
    out["miles_from_anchor"] = haversine_miles(anchor_lat, anchor_lon, out["lat"], out["lon"])
    out["ring"] = pd.cut(out["miles_from_anchor"], RING_EDGES,
                         labels=[f"{a}-{b} mi" for a, b in zip(RING_EDGES, RING_EDGES[1:])],
                         include_lowest=True)
    return out


# --- market metrics ----------------------------------------------------------
def herfindahl_index(counts) -> float:
    """HHI on location share, 0–10,000. US DOJ/FTC guidance treats <1,500 as
    unconcentrated; it is a rule of thumb here, not a regulatory finding."""
    counts = np.asarray([c for c in counts if c > 0], dtype=float)
    if counts.sum() == 0:
        return 0.0
    shares = counts / counts.sum() * 100
    return float((shares ** 2).sum())


def opportunity_table(catchment: pd.DataFrame, all_locations: pd.DataFrame,
                      outcode_geo: pd.DataFrame, anchor_lat: float, anchor_lon: float,
                      max_miles: float = CATCHMENT_MILES) -> pd.DataFrame:
    """One row per postcode district inside the catchment:
      homecare  — registered homecare agencies based there (supply)
      gp / care_homes — CQC GP practices and care homes there, used ONLY as a
                        rough proxy for how many older people live there
                        (demand). It is a proxy, not a population figure.
      homecare_per_care_home — supply against that demand proxy; low = thin cover
    """
    geo = outcode_geo.copy()
    geo["miles_from_anchor"] = haversine_miles(anchor_lat, anchor_lon, geo["lat"], geo["lon"])
    geo = geo[geo["miles_from_anchor"] <= max_miles]

    here = all_locations[all_locations["outcode"].isin(geo["outcode"])]
    table = geo.set_index("outcode")[["lat", "lon", "district", "miles_from_anchor"]].copy()
    table["homecare"] = catchment.groupby("outcode").size()
    table["gp"] = here[has_service(here, GP_SERVICE)].groupby("outcode").size()
    table["care_homes"] = here[has_service(here, *CARE_HOME_SERVICES)].groupby("outcode").size()
    table[["homecare", "gp", "care_homes"]] = table[["homecare", "gp", "care_homes"]].fillna(0).astype(int)
    table["homecare_per_care_home"] = (table["homecare"] / table["care_homes"].replace(0, np.nan)).round(2)
    return table.sort_values("miles_from_anchor")


# --- Week 3 pattern: quality gates + idempotent star-schema load --------------
@dataclass
class QualityReport:
    results: list = field(default_factory=list)

    def add(self, name, passed, blocking=True, detail=""):
        self.results.append({"check": name, "passed": bool(passed),
                             "blocking": blocking, "detail": detail})

    @property
    def blocking_failures(self):
        return [r for r in self.results if r["blocking"] and not r["passed"]]

    @property
    def warnings(self):
        return [r for r in self.results if not r["blocking"] and not r["passed"]]

    @property
    def ok_to_load(self):
        return not self.blocking_failures

    def to_frame(self):
        return pd.DataFrame(self.results)


def run_quality_checks(locations: pd.DataFrame, min_geocoded_share=0.95) -> QualityReport:
    rep = QualityReport()
    required = ["location_id", "provider_id", "location_name", "postcode", "outcode",
                "lat", "lon", "service_types"]
    missing = [c for c in required if c not in locations.columns]
    rep.add("required_columns_present", not missing, detail=f"missing={missing}")
    if missing:
        return rep

    rep.add("not_empty", len(locations) > 0, detail=f"{len(locations)} rows")
    dups = locations["location_id"][locations["location_id"].duplicated()].tolist()
    rep.add("location_id_unique", not dups, detail=f"duplicates={dups[:5]}")
    rep.add("ids_present", not locations["location_id"].apply(is_blank).any()
            and not locations["provider_id"].apply(is_blank).any())
    rep.add("all_rows_are_homecare", bool(has_service(locations, HOMECARE_SERVICE).all()),
            detail="every row must carry the 'Homecare agencies' service type")

    share = float(locations["lat"].notna().mean())
    rep.add("geocoding_coverage", share >= min_geocoded_share,
            detail=f"{share:.1%} geocoded; minimum {min_geocoded_share:.0%}")
    rep.add("all_rows_geocoded", share == 1.0, blocking=False,
            detail=f"{int(locations['lat'].isna().sum())} row(s) without coordinates")

    inbox = locations["lat"].between(49.8, 55.9) & locations["lon"].between(-6.5, 2.1)
    rep.add("coordinates_inside_great_britain", bool(inbox[locations["lat"].notna()].all()),
            detail="lat/lon must fall inside a GB bounding box")

    bad_pc = int((~locations["postcode"].apply(is_valid_uk_postcode)).sum())
    rep.add("postcodes_well_formed", bad_pc == 0, blocking=False,
            detail=f"{bad_pc} malformed postcode(s), flagged not dropped")
    return rep


SCHEMA_SQL = """
PRAGMA foreign_keys = ON;
CREATE TABLE IF NOT EXISTS dim_provider (
    provider_id   TEXT PRIMARY KEY,
    provider_name TEXT NOT NULL,
    last_loaded_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS dim_area (
    outcode   TEXT PRIMARY KEY,
    district  TEXT,
    lat       REAL,
    lon       REAL
);
CREATE TABLE IF NOT EXISTS fact_homecare_location (
    snapshot_date      TEXT NOT NULL,
    location_id        TEXT NOT NULL,
    provider_id        TEXT NOT NULL REFERENCES dim_provider(provider_id),
    outcode            TEXT NOT NULL REFERENCES dim_area(outcode),
    location_name      TEXT,
    postcode           TEXT,
    lat                REAL,
    lon                REAL,
    miles_from_anchor  REAL,
    in_catchment       INTEGER NOT NULL,
    has_website        INTEGER NOT NULL,
    latest_check       TEXT,
    specialisms        TEXT,
    PRIMARY KEY (snapshot_date, location_id)
);
CREATE TABLE IF NOT EXISTS load_audit (
    run_id TEXT PRIMARY KEY, snapshot_date TEXT NOT NULL, finished_at TEXT,
    status TEXT NOT NULL, rows_in INTEGER, rows_loaded INTEGER, notes TEXT
);
"""


def connect(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON;")   # off by default, per connection
    return conn


def load_snapshot(conn, locations: pd.DataFrame, snapshot_date: str):
    """Idempotent: dimensions upserted, this snapshot's facts deleted and
    re-inserted, all inside one transaction."""
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with conn:
        conn.executescript(SCHEMA_SQL)
        for pid, name in locations[["provider_id", "provider_name"]].drop_duplicates().values:
            conn.execute("INSERT INTO dim_provider VALUES (?,?,?) ON CONFLICT(provider_id) "
                         "DO UPDATE SET provider_name=excluded.provider_name, "
                         "last_loaded_at=excluded.last_loaded_at", (pid, name, now))
        areas = locations[["outcode", "district"]].drop_duplicates("outcode")
        for _, r in areas.iterrows():
            conn.execute("INSERT INTO dim_area (outcode, district) VALUES (?,?) "
                         "ON CONFLICT(outcode) DO UPDATE SET district=excluded.district",
                         (r["outcode"], r["district"]))
        conn.execute("DELETE FROM fact_homecare_location WHERE snapshot_date = ?", (snapshot_date,))
        rows = [(
            snapshot_date, r.location_id, r.provider_id, r.outcode, r.location_name, r.postcode,
            None if pd.isna(r.lat) else float(r.lat), None if pd.isna(r.lon) else float(r.lon),
            None if pd.isna(r.miles_from_anchor) else float(r.miles_from_anchor),
            int(bool(r.in_catchment)), int(not is_blank(r.website)),
            None if pd.isna(r.latest_check) else str(r.latest_check.date()), r.specialisms,
        ) for r in locations.itertuples()]
        conn.executemany("INSERT INTO fact_homecare_location VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)", rows)
        loaded = conn.execute("SELECT COUNT(*) FROM fact_homecare_location WHERE snapshot_date=?",
                              (snapshot_date,)).fetchone()[0]
        if loaded != len(locations):
            raise RuntimeError(f"Reconciliation failed: {len(locations)} in, {loaded} loaded")
    return loaded


def run_pipeline(conn, locations: pd.DataFrame, snapshot_date: str):
    """Checks -> gate -> load -> audit. Returns (status, QualityReport)."""
    import uuid
    report = run_quality_checks(locations)
    run_id = uuid.uuid4().hex
    with conn:
        conn.executescript(SCHEMA_SQL)
    if not report.ok_to_load:
        notes = "; ".join(f"{r['check']}: {r['detail']}" for r in report.blocking_failures)
        with conn:
            conn.execute("INSERT INTO load_audit VALUES (?,?,?,?,?,?,?)",
                         (run_id, snapshot_date, datetime.now(timezone.utc).isoformat(timespec="seconds"),
                          "BLOCKED", len(locations), 0, notes))
        return "BLOCKED", report
    loaded = load_snapshot(conn, locations, snapshot_date)
    notes = "; ".join(f"{r['check']}: {r['detail']}" for r in report.warnings)
    with conn:
        conn.execute("INSERT INTO load_audit VALUES (?,?,?,?,?,?,?)",
                     (run_id, snapshot_date, datetime.now(timezone.utc).isoformat(timespec="seconds"),
                      "LOADED", len(locations), loaded, notes))
    return "LOADED", report
