import os

import cv2
import numpy as np

from config import CONFIG
from face_service import create_face_app, get_largest_face, save_face_database


def parse_employee_folder(folder_name):
    if "_" not in folder_name:
        return folder_name, folder_name

    employee_code, full_name = folder_name.split("_", 1)
    return employee_code, full_name


def main():
    print("Dang khoi tao model InsightFace...")
    app = create_face_app()

    if not CONFIG.dataset_dir.exists():
        print(f"Khong tim thay thu muc: {CONFIG.dataset_dir}")
        return

    employee_folders = sorted(os.listdir(CONFIG.dataset_dir))
    print("--------------------------------")
    print("Danh sach thu muc nhan vien tim thay:")
    for folder in employee_folders:
        print(folder)
    print("--------------------------------")

    face_database = []

    for employee_folder in employee_folders:
        employee_path = CONFIG.dataset_dir / employee_folder
        if not employee_path.is_dir():
            continue

        employee_code, full_name = parse_employee_folder(employee_folder)
        embeddings = []

        print(f"Dang xu ly nhan vien: {employee_folder}")

        for image_path in sorted(employee_path.iterdir()):
            if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue

            img = cv2.imread(str(image_path))
            if img is None:
                print(f"Khong doc duoc anh: {image_path}")
                continue

            faces = app.get(img)
            face = get_largest_face(faces)
            if face is None:
                print(f"Khong tim thay khuon mat trong anh: {image_path}")
                continue

            embeddings.append(face.normed_embedding)
            print(f"Da them anh: {image_path}")

        if not embeddings:
            print(f"Bo qua {employee_folder}: khong co embedding hop le.")
            continue

        mean_embedding = np.mean(np.asarray(embeddings), axis=0)
        norm = np.linalg.norm(mean_embedding)
        if norm > 0:
            mean_embedding = mean_embedding / norm

        face_database.append({
            "employee_code": employee_code,
            "employee_name": employee_folder,
            "full_name": full_name,
            "embedding_count": len(embeddings),
            "embedding": mean_embedding,
        })

    if not face_database:
        print("Chua tao duoc du lieu khuon mat nao.")
        print("Hay kiem tra lai anh trong thu muc dataset.")
        return

    save_face_database(face_database)

    print("--------------------------------")
    print("Hoan thanh tao du lieu nhan dien.")
    print(f"So luong nhan vien da tao embedding: {len(face_database)}")
    print(f"Da luu vao file: {CONFIG.face_database_file}")


if __name__ == "__main__":
    main()
