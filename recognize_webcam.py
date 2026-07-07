import cv2
import pickle
import numpy as np
from insightface.app import FaceAnalysis


FACE_DATABASE_FILE = "face_database.pkl"
CAMERA_INDEX = 0
THRESHOLD = 0.45



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


def main():
    print("Dang tai du lieu khuon mat...")

    try:
        face_database = load_face_database()
    except FileNotFoundError:
        print("Khong tim thay file face_database.pkl")
        print("Hay chay create_embeddings.py truoc")
        return

    print(f"So luong embedding da tai: {len(face_database)}")

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

        cv2.imshow("Face Recognition Webcam", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()