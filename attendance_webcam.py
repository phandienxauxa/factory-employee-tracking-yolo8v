import cv2
import csv
import os
import pickle
import numpy as np
from datetime import datetime
from insightface.app import FaceAnalysis


FACE_DATABASE_FILE = "face_database.pkl"
ATTENDANCE_FILE = "attendance_logs.csv"
CAPTURE_DIR = "captures/attendance"

CAMERA_INDEX = 0
THRESHOLD = 0.45

COOLDOWN_SECONDS = 60


def load_face_database():
    with open(FACE_DATABASE_FILE, "rb") as f:
        face_database = pickle.load(f)

    return face_database


def recognize_face(face_embedding, face_database):
    best_match_name = "Unknown"
    best_score = -1

    for data in face_database:
        known_embedding = data["embedding"]

        score = np.dot(face_embedding, known_embedding)

        if score > best_score:
            best_score = score
            best_match_name = data["employee_name"]

    if best_score < THRESHOLD:
        best_match_name = "Unknown"

    return best_match_name, best_score


def create_attendance_file_if_not_exists():
    if not os.path.exists(ATTENDANCE_FILE):
        with open(ATTENDANCE_FILE, mode="w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow([
                "employee_name",
                "date",
                "time",
                "check_type",
                "confidence",
                "image_path"
            ])


def has_checked_in_today(employee_name):
    today = datetime.now().strftime("%Y-%m-%d")

    if not os.path.exists(ATTENDANCE_FILE):
        return False

    with open(ATTENDANCE_FILE, mode="r", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        for row in reader:
            if (
                row["employee_name"] == employee_name
                and row["date"] == today
                and row["check_type"] == "CHECK_IN"
            ):
                return True

    return False


def save_attendance(employee_name, confidence, frame):
    now = datetime.now()

    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")
    timestamp = now.strftime("%Y%m%d_%H%M%S")

    os.makedirs(CAPTURE_DIR, exist_ok=True)

    image_filename = f"{employee_name}_{timestamp}.jpg"
    image_path = os.path.join(CAPTURE_DIR, image_filename)

    cv2.imwrite(image_path, frame)

    with open(ATTENDANCE_FILE, mode="a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([
            employee_name,
            date_str,
            time_str,
            "CHECK_IN",
            round(float(confidence), 4),
            image_path
        ])

    print("--------------------------------")
    print(f"Da ghi diem danh: {employee_name}")
    print(f"Ngay: {date_str}")
    print(f"Gio: {time_str}")
    print(f"Do tin cay: {confidence:.2f}")
    print(f"Anh minh chung: {image_path}")


def main():
    print("Dang tai du lieu khuon mat...")

    try:
        face_database = load_face_database()
    except FileNotFoundError:
        print("Khong tim thay file face_database.pkl")
        print("Hay chay create_embeddings.py truoc")
        return

    print(f"So luong embedding da tai: {len(face_database)}")

    create_attendance_file_if_not_exists()

    print("Dang khoi tao model InsightFace...")

    app = FaceAnalysis(
        name="buffalo_l",
        providers=["CPUExecutionProvider"]
    )

    app.prepare(ctx_id=-1, det_size=(640, 640))

    cap = cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():
        print("Khong mo duoc webcam")
        return

    last_record_time = {}

    print("Da mo webcam")
    print("Nhan Q de thoat")

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Khong doc duoc hinh anh tu webcam")
            break

        faces = app.get(frame)

        for face in faces:
            x1, y1, x2, y2 = face.bbox.astype(int)

            embedding = face.normed_embedding
            name, score = recognize_face(embedding, face_database)

            if name == "Unknown":
                color = (0, 0, 255)
            else:
                color = (0, 255, 0)

                current_time = datetime.now()

                if name not in last_record_time:
                    last_record_time[name] = None

                can_record = False

                if last_record_time[name] is None:
                    can_record = True
                else:
                    elapsed = (current_time - last_record_time[name]).total_seconds()

                    if elapsed > COOLDOWN_SECONDS:
                        can_record = True

                if can_record:
                    if not has_checked_in_today(name):
                        save_attendance(name, score, frame)
                        last_record_time[name] = current_time
                    else:
                        print(f"{name} da CHECK_IN hom nay, khong ghi trung.")
                        last_record_time[name] = current_time

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                color,
                2
            )

            label = f"{name} - {score:.2f}"

            cv2.putText(
                frame,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2
            )

        cv2.imshow("Attendance Webcam", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()