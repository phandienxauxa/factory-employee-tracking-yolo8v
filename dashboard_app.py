import os
import pandas as pd
import streamlit as st


ATTENDANCE_FILE = "attendance_logs.csv"


st.set_page_config(
    page_title="He thong diem danh bang khuon mat",
    layout="wide"
)


st.title("HE THONG DIEM DANH BANG KHUON MAT")
st.write("Giao dien xem lich su diem danh cua cong nhan")


if not os.path.exists(ATTENDANCE_FILE):
    st.warning("Chua co file attendance_logs.csv")
    st.write("Hay chay attendance_webcam.py truoc de ghi du lieu diem danh.")
    st.stop()


df = pd.read_csv(ATTENDANCE_FILE)


if df.empty:
    st.warning("File attendance_logs.csv dang rong.")
    st.stop()


st.subheader("Du lieu diem danh")


col1, col2, col3 = st.columns(3)

with col1:
    selected_date = st.selectbox(
        "Chon ngay",
        options=["Tat ca"] + sorted(df["date"].unique().tolist())
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


st.write(f"So dong du lieu: {len(filtered_df)}")

st.dataframe(filtered_df, use_container_width=True)


st.subheader("Thong ke nhanh")

total_records = len(filtered_df)
total_employees = filtered_df["employee_name"].nunique()

col1, col2 = st.columns(2)

with col1:
    st.metric("So luot diem danh", total_records)

with col2:
    st.metric("So nhan vien", total_employees)


st.subheader("Anh minh chung diem danh")

for index, row in filtered_df.iterrows():
    employee_name = row["employee_name"]
    date = row["date"]
    time = row["time"]
    check_type = row["check_type"]
    confidence = row["confidence"]
    image_path = row["image_path"]

    with st.expander(f"{employee_name} - {date} {time} - {check_type}"):
        st.write(f"Nhan vien: {employee_name}")
        st.write(f"Ngay: {date}")
        st.write(f"Gio: {time}")
        st.write(f"Loai diem danh: {check_type}")
        st.write(f"Do tin cay: {confidence}")

        if os.path.exists(image_path):
            st.image(image_path, caption="Anh minh chung", width=400)
        else:
            st.warning("Khong tim thay anh minh chung.")