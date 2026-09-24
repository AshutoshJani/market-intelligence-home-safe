"""Home Safe -- Week 3: Load & Orchestrate.

Star-schema load into SQLite with:
  * idempotent re-runs (delete-then-insert per snapshot_date, inside ONE transaction),
  * data-quality checks that act as GATES (a blocking failure stops the load),
  * a load_audit table so every run -- passed or blocked -- leaves a record.

No network calls in this module. It reads Week 1 / Week 2 outputs that already
exist on disk, so the whole thing is testable with pytest and no API keys.
"""
from __future__ import annotations

import math
import re
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import date, datetime, timezone

import pandas as pd

# ---------------------------------------------------------------------------
# Constants (same lists as Weeks 1-2 -- re-check the district list before Week 4,
# Kent local government reorganisation is still in progress)
# ---------------------------------------------------------------------------
KENT_LOCAL_AUTHORITIES = [
    "Ashford", "Canterbury", "Dartford", "Dover", "Folkestone and Hythe",
    "Gravesham", "Maidstone", "Sevenoaks", "Swale", "Thanet",
    "Tonbridge and Malling", "Tunbridge Wells",
]

# CQC's four published overall ratings. A blank rating means "not yet rated",
# which is legitimate -- anything ELSE is a data problem.
VALID_RATINGS = {"Outstanding", "Good", "Requires improvement", "Inadequate"}

UK_POSTCODE_RE = re.compile(r"^[A-Z]{1,2}\d[A-Z\d]?\s?\d[A-Z]{2}$")

NOT_RATED_KEY = "NOT_RATED"

REQUIRED_LOCATION_COLUMNS = [
    "location_id", "provider_id", "location_name", "postal_code",
    "local_authority", "registration_status", "overall_rating", "overall_rating_date",
]
REQUIRED_PROVIDER_COLUMNS = ["provider_id", "provider_name"]

# Optional provider columns carried into dim_provider if present in Week 2 output.
PROVIDER_OPTIONAL_COLUMNS = [
    "ownership_type", "companies_house_number", "charity_number", "match_method",
    "match_score", "ch_company_status", "ch_date_of_creation", "ch_company_type",
]

# Thresholds -- team decisions, not facts. Document any change in the runbook.
MAX_INVALID_POSTCODE_SHARE = 0.05   # >5% malformed postcodes blocks the load
MAX_ROW_COUNT_CHANGE_SHARE = 0.20   # >20% swing vs the previous snapshot blocks


# ---------------------------------------------------------------------------
# Small pure helpers
# ---------------------------------------------------------------------------
def is_blank(value) -> bool:
    """NaN-aware blank check. NaN is truthy in Python -- never use `if value`
    on a pandas cell (the Week 2 bug)."""
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return str(value).strip() == "" or str(value).strip().lower() in {"nan", "nat", "none"}


def clean_text(value):
    return None if is_blank(value) else str(value).strip()


def is_valid_uk_postcode(value) -> bool:
    if is_blank(value):
        return False
    return bool(UK_POSTCODE_RE.match(str(value).strip().upper()))


def parse_iso_date(value):
    """Return a datetime.date, or None if blank/unparseable. Accepts
    '2024-05-01' and '2024-05-01T00:00:00' style strings."""
    if is_blank(value):
        return None
    try:
        return pd.to_datetime(str(value).strip()[:10], format="%Y-%m-%d").date()
    except (ValueError, TypeError):
        return None


def rating_period_key(rating, rating_date) -> str:
    """dim_rating_period grain = calendar month of the rating's report date.
    A location with no rating (never inspected) maps to NOT_RATED, not NULL,
    so the fact table's foreign key is always populated."""
    if is_blank(rating):
        return NOT_RATED_KEY
    d = parse_iso_date(rating_date)
    return NOT_RATED_KEY if d is None else f"{d.year:04d}-{d.month:02d}"


# ---------------------------------------------------------------------------
# Data-quality checks
# ---------------------------------------------------------------------------
@dataclass
class CheckResult:
    name: str
    passed: bool
    blocking: bool
    detail: str = ""


@dataclass
class QualityReport:
    results: list = field(default_factory=list)

    def add(self, name, passed, blocking=True, detail=""):
        self.results.append(CheckResult(name, bool(passed), blocking, detail))

    @property
    def blocking_failures(self):
        return [r for r in self.results if r.blocking and not r.passed]

    @property
    def warnings(self):
        return [r for r in self.results if not r.blocking and not r.passed]

    @property
    def ok_to_load(self) -> bool:
        return not self.blocking_failures

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([r.__dict__ for r in self.results])


def run_quality_checks(locations: pd.DataFrame, providers: pd.DataFrame,
                       snapshot_date: date, previous_row_count: int | None = None,
                       allow_row_count_change: bool = False) -> QualityReport:
    """Run every check BEFORE touching the database. Blocking failures stop the
    load; warnings are recorded in load_audit but let the load continue."""
    rep = QualityReport()

    # 1. Schema -- if columns are missing, nothing else is meaningful.
    missing_loc = [c for c in REQUIRED_LOCATION_COLUMNS if c not in locations.columns]
    missing_prov = [c for c in REQUIRED_PROVIDER_COLUMNS if c not in providers.columns]
    rep.add("required_columns_present", not missing_loc and not missing_prov,
            detail=f"missing locations={missing_loc} providers={missing_prov}")
    if missing_loc or missing_prov:
        return rep

    # 2. Not empty
    rep.add("locations_not_empty", len(locations) > 0, detail=f"{len(locations)} rows")

    # 3. Keys present
    blank_loc_ids = int(locations["location_id"].apply(is_blank).sum())
    blank_prov_ids = int(locations["provider_id"].apply(is_blank).sum())
    rep.add("location_and_provider_ids_present", blank_loc_ids == 0 and blank_prov_ids == 0,
            detail=f"blank location_id={blank_loc_ids}, blank provider_id={blank_prov_ids}")

    # 4. Uniqueness (the spec's "no duplicate IDs" check, applied at both grains)
    dup_loc = locations["location_id"][locations["location_id"].duplicated()].unique().tolist()
    rep.add("location_id_unique", not dup_loc, detail=f"duplicates: {dup_loc[:10]}")
    dup_prov = providers["provider_id"][providers["provider_id"].duplicated()].unique().tolist()
    rep.add("provider_id_unique", not dup_prov, detail=f"duplicates: {dup_prov[:10]}")

    # 5. Referential integrity -- every location's provider must exist in dim_provider
    orphans = sorted(set(locations["provider_id"].dropna()) - set(providers["provider_id"].dropna()))
    rep.add("every_location_has_a_provider", not orphans, detail=f"orphan provider_ids: {orphans[:10]}")

    # 6. Rating domain -- blank is fine (not yet rated), an unknown label is not
    bad_ratings = sorted({str(r) for r in locations["overall_rating"]
                          if not is_blank(r) and str(r).strip() not in VALID_RATINGS})
    rep.add("overall_rating_in_allowed_values", not bad_ratings, detail=f"unexpected: {bad_ratings}")

    # 7. Rating dates must parse and not be in the future
    rated = locations[~locations["overall_rating"].apply(is_blank)]
    parsed = rated["overall_rating_date"].apply(parse_iso_date)
    unparseable = int(parsed.isna().sum())
    future = int(sum(1 for d in parsed if d is not None and d > snapshot_date))
    rep.add("rating_dates_valid_and_not_future", unparseable == 0 and future == 0,
            detail=f"unparseable={unparseable}, after snapshot_date={future}")

    # 8. Postcodes -- a few bad ones is a warning; lots means something upstream broke
    n = max(len(locations), 1)
    invalid_pc = int((~locations["postal_code"].apply(is_valid_uk_postcode)).sum())
    share = invalid_pc / n
    rep.add("postcode_invalid_share_within_threshold", share <= MAX_INVALID_POSTCODE_SHARE,
            detail=f"{invalid_pc} invalid ({share:.1%}); limit {MAX_INVALID_POSTCODE_SHARE:.0%}")
    rep.add("all_postcodes_valid", invalid_pc == 0, blocking=False,
            detail=f"{invalid_pc} invalid postcode(s) loaded and flagged")

    # 9. Local authority outside the 12 districts -- flag, never drop (Week 2 rule)
    la_lookup = {la.lower() for la in KENT_LOCAL_AUTHORITIES}
    outside = sorted({str(v) for v in locations["local_authority"]
                      if is_blank(v) or str(v).strip().lower() not in la_lookup})
    rep.add("local_authority_in_kent_list", not outside, blocking=False,
            detail=f"outside list: {outside}")

    # 10. Volume drift vs the last successful snapshot
    if previous_row_count:
        change = abs(len(locations) - previous_row_count) / previous_row_count
        rep.add("row_count_change_within_threshold",
                change <= MAX_ROW_COUNT_CHANGE_SHARE or allow_row_count_change,
                detail=(f"{previous_row_count} -> {len(locations)} ({change:.0%}); "
                        f"limit {MAX_ROW_COUNT_CHANGE_SHARE:.0%}"
                        + ("; OVERRIDDEN by operator" if allow_row_count_change else "")))
    return rep


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------
SCHEMA_SQL = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS dim_area (
    area_key          INTEGER PRIMARY KEY,
    local_authority   TEXT NOT NULL UNIQUE,
    is_kent_district  INTEGER NOT NULL CHECK (is_kent_district IN (0, 1))
);

CREATE TABLE IF NOT EXISTS dim_provider (
    provider_id            TEXT PRIMARY KEY,
    provider_name          TEXT NOT NULL,
    ownership_type         TEXT,
    companies_house_number TEXT,
    charity_number         TEXT,
    match_method           TEXT,
    match_score            REAL,
    ch_company_status      TEXT,
    ch_date_of_creation    TEXT,
    ch_company_type        TEXT,
    last_loaded_at         TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS dim_rating_period (
    rating_period_key  TEXT PRIMARY KEY,      -- 'YYYY-MM' or 'NOT_RATED'
    year               INTEGER,
    month              INTEGER,
    quarter            INTEGER
);

CREATE TABLE IF NOT EXISTS fact_location_snapshot (
    snapshot_date        TEXT NOT NULL,       -- the day this pipeline run observed CQC
    location_id          TEXT NOT NULL,
    provider_id          TEXT NOT NULL REFERENCES dim_provider(provider_id),
    area_key             INTEGER NOT NULL REFERENCES dim_area(area_key),
    rating_period_key    TEXT NOT NULL REFERENCES dim_rating_period(rating_period_key),
    location_name        TEXT,
    postal_code          TEXT,
    postcode_valid       INTEGER NOT NULL,
    registration_status  TEXT,
    overall_rating       TEXT,
    overall_rating_date  TEXT,
    is_rated             INTEGER NOT NULL,
    PRIMARY KEY (snapshot_date, location_id)
);

CREATE TABLE IF NOT EXISTS load_audit (
    run_id          TEXT PRIMARY KEY,
    snapshot_date   TEXT NOT NULL,
    started_at      TEXT NOT NULL,
    finished_at     TEXT,
    status          TEXT NOT NULL,            -- LOADED | BLOCKED | FAILED
    rows_in         INTEGER,
    rows_loaded     INTEGER,
    blocking_failed TEXT,
    warnings        TEXT
);
"""


def connect(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON;")  # per-connection in SQLite, off by default
    return conn


def create_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA_SQL)


def previous_snapshot_row_count(conn, snapshot_date: date):
    """Row count of the most recent LOADED snapshot strictly before this one."""
    row = conn.execute(
        """SELECT rows_loaded FROM load_audit
           WHERE status = 'LOADED' AND snapshot_date < ?
           ORDER BY snapshot_date DESC, finished_at DESC LIMIT 1""",
        (snapshot_date.isoformat(),),
    ).fetchone()
    return row[0] if row else None


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
def _upsert_areas(conn, local_authorities):
    kent = {la.lower(): la for la in KENT_LOCAL_AUTHORITIES}
    for raw in sorted({clean_text(v) or "UNKNOWN" for v in local_authorities}):
        canonical = kent.get(raw.lower(), raw)
        conn.execute(
            """INSERT INTO dim_area (local_authority, is_kent_district) VALUES (?, ?)
               ON CONFLICT(local_authority) DO NOTHING""",
            (canonical, int(raw.lower() in kent)),
        )
    return {name.lower(): key for key, name in
            conn.execute("SELECT area_key, local_authority FROM dim_area")}


def _upsert_providers(conn, providers, loaded_at):
    cols = ["provider_id", "provider_name"] + PROVIDER_OPTIONAL_COLUMNS + ["last_loaded_at"]
    placeholders = ", ".join("?" for _ in cols)
    updates = ", ".join(f"{c} = excluded.{c}" for c in cols if c != "provider_id")
    sql = (f"INSERT INTO dim_provider ({', '.join(cols)}) VALUES ({placeholders}) "
           f"ON CONFLICT(provider_id) DO UPDATE SET {updates}")
    for _, r in providers.iterrows():
        values = [clean_text(r["provider_id"]), clean_text(r["provider_name"])]
        for c in PROVIDER_OPTIONAL_COLUMNS:
            v = r[c] if c in providers.columns else None
            if c == "match_score":
                values.append(None if is_blank(v) else float(v))
            else:
                values.append(clean_text(v))
        values.append(loaded_at)
        conn.execute(sql, values)


def _upsert_rating_periods(conn, keys):
    for k in sorted(set(keys)):
        if k == NOT_RATED_KEY:
            row = (k, None, None, None)
        else:
            y, m = int(k[:4]), int(k[5:7])
            row = (k, y, m, (m - 1) // 3 + 1)
        conn.execute("INSERT INTO dim_rating_period VALUES (?, ?, ?, ?) "
                     "ON CONFLICT(rating_period_key) DO NOTHING", row)


def load_snapshot(conn, locations, providers, snapshot_date: date):
    """Idempotent load of ONE snapshot. Everything happens in a single
    transaction: dims upserted, this snapshot_date's fact rows deleted, then
    re-inserted. If anything raises, the whole thing rolls back -- the database
    is never left half-loaded."""
    loaded_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    snap = snapshot_date.isoformat()
    with conn:  # sqlite3: commit on success, rollback on exception
        area_keys = _upsert_areas(conn, locations["local_authority"])
        _upsert_providers(conn, providers, loaded_at)
        rp_keys = [rating_period_key(r, d) for r, d in
                   zip(locations["overall_rating"], locations["overall_rating_date"])]
        _upsert_rating_periods(conn, rp_keys)

        conn.execute("DELETE FROM fact_location_snapshot WHERE snapshot_date = ?", (snap,))
        rows = []
        for (_, r), rp in zip(locations.iterrows(), rp_keys):
            la = (clean_text(r["local_authority"]) or "UNKNOWN").lower()
            pc = clean_text(r["postal_code"])
            rating = clean_text(r["overall_rating"])
            rdate = parse_iso_date(r["overall_rating_date"])
            rows.append((
                snap, clean_text(r["location_id"]), clean_text(r["provider_id"]),
                area_keys[la], rp, clean_text(r["location_name"]),
                pc.upper() if pc else None, int(is_valid_uk_postcode(pc)),
                clean_text(r["registration_status"]), rating,
                rdate.isoformat() if rdate else None, int(rating is not None),
            ))
        conn.executemany(
            "INSERT INTO fact_location_snapshot VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", rows)
        loaded = conn.execute("SELECT COUNT(*) FROM fact_location_snapshot WHERE snapshot_date = ?",
                              (snap,)).fetchone()[0]
        if loaded != len(locations):  # post-load reconciliation, still inside the transaction
            raise RuntimeError(f"Reconciliation failed: {len(locations)} in, {loaded} loaded")
    return loaded


def _write_audit(conn, run_id, snapshot_date, started_at, status, rows_in,
                 rows_loaded, report: QualityReport | None, message=""):
    blocking = "; ".join(f"{r.name}: {r.detail}" for r in report.blocking_failures) if report else ""
    warns = "; ".join(f"{r.name}: {r.detail}" for r in report.warnings) if report else ""
    if message:
        blocking = (blocking + "; " if blocking else "") + message
    with conn:
        conn.execute(
            "INSERT INTO load_audit VALUES (?,?,?,?,?,?,?,?,?)",
            (run_id, snapshot_date.isoformat(), started_at,
             datetime.now(timezone.utc).isoformat(timespec="seconds"),
             status, rows_in, rows_loaded, blocking, warns))


def run_pipeline(conn, locations, providers, snapshot_date: date,
                 allow_row_count_change: bool = False):
    """Checks -> gate -> load -> audit. Returns (status, QualityReport)."""
    create_schema(conn)
    run_id = uuid.uuid4().hex
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    prev = previous_snapshot_row_count(conn, snapshot_date)
    report = run_quality_checks(locations, providers, snapshot_date, prev, allow_row_count_change)

    if not report.ok_to_load:
        _write_audit(conn, run_id, snapshot_date, started, "BLOCKED", len(locations), 0, report)
        return "BLOCKED", report
    try:
        loaded = load_snapshot(conn, locations, providers, snapshot_date)
    except Exception as e:  # recorded, then re-raised so a scheduler sees a failure
        _write_audit(conn, run_id, snapshot_date, started, "FAILED", len(locations), 0,
                     report, message=f"{type(e).__name__}: {e}")
        raise
    _write_audit(conn, run_id, snapshot_date, started, "LOADED", len(locations), loaded, report)
    return "LOADED", report


# ---------------------------------------------------------------------------
# A Week 4 preview query -- proves the star schema answers the business question
# ---------------------------------------------------------------------------
DENSITY_BY_AREA_SQL = """
SELECT a.local_authority,
       COUNT(DISTINCT f.provider_id)                                   AS providers,
       COUNT(*)                                                        AS locations,
       SUM(CASE WHEN f.overall_rating = 'Outstanding' THEN 1 ELSE 0 END)          AS outstanding,
       SUM(CASE WHEN f.overall_rating = 'Good' THEN 1 ELSE 0 END)                 AS good,
       SUM(CASE WHEN f.overall_rating = 'Requires improvement' THEN 1 ELSE 0 END) AS requires_improvement,
       SUM(CASE WHEN f.overall_rating = 'Inadequate' THEN 1 ELSE 0 END)           AS inadequate,
       SUM(CASE WHEN f.is_rated = 0 THEN 1 ELSE 0 END)                            AS not_yet_rated,
       -- NULL = NULL is NULL in SQL, so a bare SUM(rating = 'Good') returns NULL for a
       -- district whose only locations are unrated. CASE ... ELSE 0 avoids that.
       ROUND(1.0 * SUM(CASE WHEN f.overall_rating IN ('Outstanding','Good') THEN 1 ELSE 0 END)
             / NULLIF(SUM(f.is_rated), 0), 2)                                     AS good_or_better_share
FROM fact_location_snapshot f
JOIN dim_area a ON a.area_key = f.area_key
WHERE f.snapshot_date = (SELECT MAX(snapshot_date) FROM fact_location_snapshot)
  AND f.registration_status = 'Registered'
GROUP BY a.local_authority
ORDER BY locations DESC;
"""
