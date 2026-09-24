"""Home Safe -- scheduled entry point for the Week 3 load.

This is what cron / Windows Task Scheduler / GitHub Actions runs. A notebook is
for exploring; a scheduler needs a plain script with a meaningful exit code.

Exit codes:
  0  loaded successfully
  2  blocked by a data-quality gate (database unchanged)
  1  unexpected failure (database rolled back)

Example:
  python run_pipeline.py --locations ../Week_1/home_safe_kent_personal_care_locations.csv \
                         --providers ../Week_2/home_safe_kent_providers_enriched.parquet \
                         --db home_safe_kent.db
"""
import argparse
import logging
import sys
from datetime import date
from pathlib import Path

import pandas as pd

import load_week3 as lw


def read_table(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Input not found: {p.resolve()}")
    if p.suffix.lower() == ".parquet":
        return pd.read_parquet(p)  # needs pyarrow
    # dtype=str keeps IDs like '00000006' and '1-000001' exactly as written
    return pd.read_csv(p, dtype=str, keep_default_na=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Home Safe Week 3 load")
    ap.add_argument("--locations", required=True, help="Week 1 locations CSV")
    ap.add_argument("--providers", required=True, help="Week 2 enriched providers (.parquet or .csv)")
    ap.add_argument("--db", default="home_safe_kent.db")
    ap.add_argument("--snapshot-date", default=date.today().isoformat(),
                    help="YYYY-MM-DD; defaults to today. Re-using a date REPLACES that snapshot.")
    ap.add_argument("--allow-row-count-change", action="store_true",
                    help="Operator override for the volume-drift gate. Record why in the runbook log.")
    ap.add_argument("--log-file", default="pipeline.log")
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(args.log_file), logging.StreamHandler(sys.stdout)],
    )
    log = logging.getLogger("home_safe")

    try:
        snapshot = date.fromisoformat(args.snapshot_date)
        locations = read_table(args.locations)
        providers = read_table(args.providers)
        conn = lw.connect(args.db)
        try:
            status, report = lw.run_pipeline(conn, locations, providers, snapshot,
                                             allow_row_count_change=args.allow_row_count_change)
        finally:
            conn.close()
    except Exception:
        log.exception("Pipeline FAILED -- database transaction rolled back")
        return 1

    for r in report.results:
        level = logging.INFO if r.passed else (logging.ERROR if r.blocking else logging.WARNING)
        log.log(level, "check %-42s %s  %s", r.name, "PASS" if r.passed else "FAIL", r.detail)

    if status == "BLOCKED":
        log.error("Load BLOCKED by data-quality gate for snapshot %s -- nothing written", snapshot)
        return 2
    log.info("Load complete for snapshot %s (%d locations)", snapshot, len(locations))
    return 0


if __name__ == "__main__":
    sys.exit(main())
