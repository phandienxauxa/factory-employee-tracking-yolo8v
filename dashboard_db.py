import os
import time
import sqlite3
from datetime import datetime

import pandas as pd
import streamlit as st

from config import CONFIG

def load_attendance_data():
    conn = sqlite3.connect(CONFIG.database_file)

    query = """
        SELECT *
        FROM attendance_logs
        ORDER BY date DESC, time DESC
    """

    df = pd.read_sql_query(query, conn)

    conn.close()

    return df


def rerun_app():
    if hasattr(st, "rerun"):
        st.rerun()
    else:
        st.experimental_rerun()


st.set_page_config(
    page_title="Dashboard diem danh bang khuon mat",
    layout="wide"
)


st.title("DASHBOARD DIEM DANH BANG KHUON MAT")
st.write("Giao dien quan ly du lieu diem danh tu SQLite database")


# =========================
# TU DONG CAP NHAT
# =========================

col_refresh_1, col_refresh_2 = st.columns([1, 2])

with col_refresh_1:
    auto_refresh = st.checkbox(
        "Tu dong cap nhat",
        value=True
    )

with col_refresh_2:
    refresh_seconds = st.selectbox(
        "Thoi gian cap nhat",
        options=[3, 5, 10, 30],
        index=1
    )


st.write(
    "Lan cap nhat gan nhat:",
    datetime.now().strftime("%Y-%m-%d %H:%M:%S")
)


# =========================
# KIEM TRA DATABASE
# =========================

if not CONFIG.database_file.exists():
    st.error("Khong tim thay file database.db")
    st.write("Hay chay init_database.py truoc.")
    
    if auto_refresh:
        time.sleep(refresh_seconds)
        rerun_app()

    st.stop()


df = load_attendance_data()


if df.empty:
    st.warning("Chua co du lieu diem danh trong database.")
    st.write("Hay chay attendance_webcam_db.py de ghi du lieu diem danh.")

    if auto_refresh:
        time.sleep(refresh_seconds)
        rerun_app()

    st.stop()


# =========================
# BO LOC DU LIEU
# =========================

st.subheader("Bo loc du lieu")

col1, col2, col3 = st.columns(3)

with col1:
    selected_date = st.selectbox(
        "Chon ngay",
        options=["Tat ca"] + sorted(df["date"].unique().tolist(), reverse=True)
    )

with col2:
    selected_employee = st.selectbox(
        "Chon nhan vien",
        options=["Tat ca"] + sorted(df["employee_name"].unique().tolist())
    )

with col3:
    selected_check_type = st.selectbox(
        "Loai diem danh",
        options=["Tat ca"] + sorted(df["check_type"].unique().tolist())
    )


filtered_df = df.copy()

if selected_date != "Tat ca":
    filtered_df = filtered_df[filtered_df["date"] == selected_date]

if selected_employee != "Tat ca":
    filtered_df = filtered_df[filtered_df["employee_name"] == selected_employee]

if selected_check_type != "Tat ca":
    filtered_df = filtered_df[filtered_df["check_type"] == selected_check_type]


# =========================
# THONG KE NHANH
# =========================

st.subheader("Thong ke nhanh")

total_records = len(filtered_df)
total_employees = filtered_df["employee_name"].nunique()
total_check_in = len(filtered_df[filtered_df["check_type"] == "CHECK_IN"])
total_check_out = len(filtered_df[filtered_df["check_type"] == "CHECK_OUT"])

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("So luot diem danh", total_records)

with col2:
    st.metric("So nhan vien", total_employees)

with col3:
    st.metric("So luot CHECK_IN", total_check_in)

with col4:
    st.metric("So luot CHECK_OUT", total_check_out)


# =========================
# BANG DU LIEU
# =========================

st.subheader("Bang du lieu diem danh")

st.dataframe(
    filtered_df,
    use_container_width=True
)


# =========================
# TAI DU LIEU CSV
# =========================

st.subheader("Tai du lieu diem danh")

csv_data = filtered_df.to_csv(index=False).encode("utf-8-sig")

st.download_button(
    label="Tai file CSV",
    data=csv_data,
    file_name="attendance_export.csv",
    mime="text/csv"
)


# =========================
# ANH MINH CHUNG
# =========================

st.subheader("Anh minh chung diem danh")

for index, row in filtered_df.iterrows():
    employee_name = row["employee_name"]
    date = row["date"]
    time_value = row["time"]
    check_type = row["check_type"]
    confidence = row["confidence"]
    image_path = row["image_path"]

    title = f"{employee_name} - {date} {time_value} - {check_type}"

    with st.expander(title):
        col1, col2 = st.columns([1, 2])

        with col1:
            st.write(f"Nhan vien: {employee_name}")
            st.write(f"Ngay: {date}")
            st.write(f"Gio: {time_value}")
            st.write(f"Loai diem danh: {check_type}")
            st.write(f"Do tin cay: {confidence}")

        with col2:
            image_path_obj = CONFIG.database_file.parent / image_path if image_path and not os.path.isabs(image_path) else image_path
            if image_path and os.path.exists(image_path_obj):
                st.image(
                    image_path_obj,
                    caption="Anh minh chung",
                    width=450
                )
            else:
                st.warning("Khong tim thay anh minh chung.")


# =========================
# TU DONG RERUN
# =========================

if auto_refresh:
    time.sleep(refresh_seconds)
    rerun_app()
