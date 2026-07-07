import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime

import cv2

from config import CONFIG


@contextmanager
def connect_db():
    conn = sqlite3.connect(CONFIG.database_file)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def create_or_migrate_database():
    with connect_db() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS employees (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                employee_code TEXT UNIQUE NOT NULL,
                full_name TEXT NOT NULL,
                department TEXT,
                position TEXT,
                status TEXT DEFAULT 'active'
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS attendance_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                employee_name TEXT NOT NULL,
                date TEXT NOT NULL,
                time TEXT NOT NULL,
                check_type TEXT NOT NULL,
                confidence REAL,
                image_path TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        ensure_column(cursor, "attendance_logs", "employee_code", "TEXT")
        ensure_column(cursor, "attendance_logs", "source", "TEXT DEFAULT 'face'")
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_attendance_employee_date
            ON attendance_logs(employee_code, employee_name, date)
        """)


def ensure_column(cursor, table_name, column_name, column_type):
    cursor.execute(f"PRAGMA table_info({table_name})")
    columns = {row[1] for row in cursor.fetchall()}
    if column_name not in columns:
        cursor.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_type}")


def get_today_records(employee_name, employee_code=None):
    today = datetime.now().strftime("%Y-%m-%d")
    with connect_db() as conn:
        cursor = conn.cursor()
        if employee_code:
            cursor.execute("""
                SELECT check_type, date, time
                FROM attendance_logs
                WHERE (employee_code = ? OR employee_name = ?)
                AND date = ?
                ORDER BY id ASC
            """, (employee_code, employee_name, today))
        else:
            cursor.execute("""
                SELECT check_type, date, time
                FROM attendance_logs
                WHERE employee_name = ?
                AND date = ?
                ORDER BY id ASC
            """, (employee_name, today))
        return cursor.fetchall()


def decide_check_type(employee_name, employee_code=None):
    records = get_today_records(employee_name, employee_code)
    check_in_datetime = None
    has_check_out = False

    for record in records:
        if record["check_type"] == "CHECK_IN":
            check_in_datetime = datetime.strptime(
                record["date"] + " " + record["time"],
                "%Y-%m-%d %H:%M:%S",
            )
        elif record["check_type"] == "CHECK_OUT":
            has_check_out = True

    if check_in_datetime is None:
        return "CHECK_IN"

    if not has_check_out:
        elapsed_seconds = (datetime.now() - check_in_datetime).total_seconds()
        if elapsed_seconds >= CONFIG.min_seconds_between_checkin_checkout:
            return "CHECK_OUT"

        remaining = CONFIG.min_seconds_between_checkin_checkout - elapsed_seconds
        print(f"{employee_name} chua du thoi gian CHECK_OUT. Con {remaining:.0f} giay.")

    return None


def save_attendance(employee_name, confidence, frame, check_type, employee_code=None):
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")
    timestamp = now.strftime("%Y%m%d_%H%M%S")

    CONFIG.attendance_capture_dir.mkdir(parents=True, exist_ok=True)
    safe_name = employee_name.replace(os.sep, "_").replace(" ", "_")
    image_filename = f"{safe_name}_{check_type}_{timestamp}.jpg"
    image_path = CONFIG.attendance_capture_dir / image_filename
    cv2.imwrite(str(image_path), frame)

    with connect_db() as conn:
        conn.execute("""
            INSERT INTO attendance_logs (
                employee_code,
                employee_name,
                date,
                time,
                check_type,
                confidence,
                image_path
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            employee_code,
            employee_name,
            date_str,
            time_str,
            check_type,
            round(float(confidence), 4),
            str(image_path),
        ))

    print("--------------------------------")
    print(f"Da ghi diem danh: {employee_name} - {check_type}")
    print(f"Ngay: {date_str}")
    print(f"Gio: {time_str}")
    print(f"Do tin cay: {confidence:.2f}")
    print(f"Anh minh chung: {image_path}")
