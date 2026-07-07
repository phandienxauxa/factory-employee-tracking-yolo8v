from datetime import datetime

import cv2

from config import CONFIG
from database_service import (
    create_or_migrate_database,
    decide_check_type,
    save_attendance,
)
from face_service import create_face_app, load_face_database, recognize_face


def can_record_now(name, current_time, last_record_time):
    last_time = last_record_time.get(name)
    if last_time is None:
        return True

    elapsed = (current_time - last_time).total_seconds()
    return elapsed > CONFIG.cooldown_seconds


def main():
    if not CONFIG.face_database_file.exists():
        print("Khong tim thay face_database.pkl")
        print("Hay chay file create_embeddings.py truoc")
        return

    create_or_migrate_database()

    print("Dang tai du lieu khuon mat...")
    face_database = load_face_database()
    print(f"So luong nhan vien da tai: {len(face_database)}")

    print("Dang khoi tao model InsightFace...")
    app = create_face_app()

    cap = cv2.VideoCapture(CONFIG.camera_index)
    if not cap.isOpened():
        print("Khong mo duoc webcam")
        return

    last_record_time = {}

    print("Da mo webcam")
    print("Nhan Q de thoat")
    print(
        "CHECK_IN truoc, sau "
        f"{CONFIG.min_seconds_between_checkin_checkout} giay moi cho CHECK_OUT"
    )

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Khong doc duoc hinh anh tu webcam")
            break

        faces = app.get(frame)

        for face in faces:
            x1, y1, x2, y2 = face.bbox.astype(int)
            match = recognize_face(face.normed_embedding, face_database)

            name = match["employee_name"]
            score = match["score"]
            color = (0, 255, 0) if match["is_known"] else (0, 0, 255)

            if match["is_known"]:
                current_time = datetime.now()
                if can_record_now(name, current_time, last_record_time):
                    check_type = decide_check_type(name, match["employee_code"])
                    if check_type is not None:
                        save_attendance(
                            name,
                            score,
                            frame,
                            check_type,
                            employee_code=match["employee_code"],
                        )
                    last_record_time[name] = current_time

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame,
                f"{name} - {score:.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2,
            )

        cv2.imshow("Attendance Webcam DB", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
