from __future__ import annotations

import sqlite3
from contextlib import closing
from datetime import datetime
from pathlib import Path

import pandas as pd


ATTENDANCE_CONCERNS = {
    "Academic confidence": "Offer a study-skills or academic-advising appointment.",
    "Financial strain": "Offer a confidential referral to the appropriate financial-support service.",
    "Wellbeing concern": "Offer a voluntary conversation and, if requested, a referral to qualified support.",
    "Care responsibilities": "Discuss available scheduling flexibility and relevant student services.",
    "Housing or transport": "Offer a referral to relevant housing, transport, or student-support services.",
    "Learning support requested": "Connect the student with the relevant learning-support service.",
    "Other staff-reported concern": "Ask what support the student would find useful and agree on a follow-up.",
}


def initialize_store(db_path: Path) -> None:
    with closing(sqlite3.connect(db_path, timeout=5)) as connection:
        with connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS support_events (
                    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    case_ref TEXT NOT NULL,
                    recorded_at TEXT NOT NULL,
                    attendance_pct REAL,
                    concerns TEXT NOT NULL,
                    intervention TEXT NOT NULL,
                    status TEXT NOT NULL,
                    notes TEXT NOT NULL
                )
                """
            )


def add_support_event(
    db_path: Path,
    case_ref: str,
    attendance_pct: float | None,
    concerns: list[str],
    intervention: str,
    status: str,
    notes: str,
) -> None:
    normalized_case_ref = case_ref.strip()
    if not normalized_case_ref or len(normalized_case_ref) > 50:
        raise ValueError("Enter a pseudonymous case reference of 1 to 50 characters.")
    if attendance_pct is not None and not 0 <= attendance_pct <= 100:
        raise ValueError("Attendance must be between 0 and 100 percent.")
    if len(notes) > 1000:
        raise ValueError("Notes must be 1,000 characters or fewer.")

    with closing(sqlite3.connect(db_path, timeout=5)) as connection:
        with connection:
            connection.execute(
                """
                INSERT INTO support_events
                    (case_ref, recorded_at, attendance_pct, concerns, intervention, status, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    normalized_case_ref,
                    datetime.now().astimezone().isoformat(timespec="seconds"),
                    attendance_pct,
                    "; ".join(concerns),
                    intervention,
                    status,
                    notes.strip(),
                ),
            )


def get_support_events(
    db_path: Path, case_ref: str | None = None
) -> pd.DataFrame:
    with closing(sqlite3.connect(db_path, timeout=5)) as connection:
        if case_ref:
            events = pd.read_sql_query(
                """
                SELECT recorded_at, case_ref, attendance_pct, concerns,
                       intervention, status, notes
                FROM support_events
                WHERE case_ref = ?
                ORDER BY recorded_at
                """,
                connection,
                params=(case_ref.strip(),),
            )
        else:
            events = pd.read_sql_query(
                """
                SELECT recorded_at, case_ref, attendance_pct, concerns,
                       intervention, status, notes
                FROM support_events
                ORDER BY recorded_at DESC
                """,
                connection,
            )
    return events


def attendance_policy_flag(
    attendance_pct: float | None, threshold_pct: float
) -> bool:
    if not 0 <= threshold_pct <= 100:
        raise ValueError("Attendance threshold must be between 0 and 100 percent.")
    if attendance_pct is not None and not 0 <= attendance_pct <= 100:
        raise ValueError("Attendance must be between 0 and 100 percent.")
    return attendance_pct is not None and attendance_pct < threshold_pct


def suggested_support(concerns: list[str]) -> list[str]:
    unknown_concerns = sorted(set(concerns) - set(ATTENDANCE_CONCERNS))
    if unknown_concerns:
        raise ValueError("Unrecognized staff-reported factors: " + ", ".join(unknown_concerns))
    suggestions = [ATTENDANCE_CONCERNS[item] for item in concerns]
    return suggestions or [
        "No concern was selected. Offer routine check-ins and ask whether support is wanted."
    ]
