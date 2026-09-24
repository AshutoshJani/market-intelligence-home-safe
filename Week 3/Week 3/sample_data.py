"""Small, clearly-labelled SAMPLE inputs shaped exactly like the real Week 1 and
Week 2 outputs (same column names). Used when the real files aren't present and
by the tests. Numbers produced from these are illustrative only."""
import numpy as np
import pandas as pd


def sample_providers() -> pd.DataFrame:
    return pd.DataFrame([
        {"provider_id": "1-000001", "provider_name": "Home Safe Domiciliary Care Ltd",
         "ownership_type": "Organisation", "companies_house_number": "00000006",
         "charity_number": np.nan, "match_method": "direct_number", "match_score": 100.0,
         "ch_company_status": "active", "ch_date_of_creation": "2012-04-02",
         "ch_company_type": "ltd"},
        {"provider_id": "1-000002", "provider_name": "Kent Homecare Services Limited",
         "ownership_type": "Organisation", "companies_house_number": np.nan,
         "charity_number": np.nan, "match_method": "fuzzy_name", "match_score": 92.0,
         "ch_company_status": "active", "ch_date_of_creation": "2014-11-19",
         "ch_company_type": "ltd"},
        {"provider_id": "1-000003", "provider_name": "Garden of England Care Trust",
         "ownership_type": "Organisation", "companies_house_number": np.nan,
         "charity_number": "1123456", "match_method": "charity_no_ch_expected",
         "match_score": np.nan, "ch_company_status": np.nan, "ch_date_of_creation": np.nan,
         "ch_company_type": np.nan},
        {"provider_id": "1-000004", "provider_name": "A. Patel (Personal Care)",
         "ownership_type": "Individual", "companies_house_number": np.nan,
         "charity_number": np.nan, "match_method": "fuzzy_no_match_above_threshold",
         "match_score": 41.0, "ch_company_status": np.nan, "ch_date_of_creation": np.nan,
         "ch_company_type": np.nan},
    ])


def sample_locations() -> pd.DataFrame:
    rows = [
        ("1-100001", "1-000001", "Home Safe Ashford", "TN23 1AB", "Ashford", "Registered", "Good", "2024-05-14"),
        ("1-100002", "1-000001", "Home Safe Canterbury", "CT1 2AB", "Canterbury", "Registered", "Outstanding", "2023-11-02"),
        ("1-100003", "1-000002", "Kent Homecare Canterbury", "CT2 7AB", "Canterbury", "Registered", "Requires improvement", "2025-02-20"),
        ("1-100004", "1-000002", "Kent Homecare Thanet", "CT9 1AB", "Thanet", "Registered", "Good", "2025-06-30"),
        ("1-100005", "1-000003", "Garden of England Maidstone", "ME14 1AB", "Maidstone", "Registered", "Good", "2022-08-09"),
        ("1-100006", "1-000004", "A. Patel Dartford", "DA1 1AB", "Dartford", "Registered", np.nan, np.nan),   # never inspected
        ("1-100007", "1-000003", "Garden of England Swale", "ME10 3AB", "Swale", "Registered", "Inadequate", "2026-01-15"),
        ("1-100008", "1-000002", "Kent Homecare Dover", "CT16 1AB", "Dover", "Deregistered", "Good", "2021-03-03"),
    ]
    cols = ["location_id", "provider_id", "location_name", "postal_code", "local_authority",
            "registration_status", "overall_rating", "overall_rating_date"]
    return pd.DataFrame(rows, columns=cols)
