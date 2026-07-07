import pickle
from pathlib import Path

import numpy as np
from insightface.app import FaceAnalysis

from config import CONFIG


def create_face_app():
    app = FaceAnalysis(
        name=CONFIG.insightface_model_name,
        providers=list(CONFIG.insightface_providers),
    )
    app.prepare(ctx_id=-1, det_size=CONFIG.insightface_det_size)
    return app


def get_largest_face(faces):
    if not faces:
        return None

    def area(face):
        x1, y1, x2, y2 = face.bbox
        return (x2 - x1) * (y2 - y1)

    return max(faces, key=area)


def load_face_database(path: Path = CONFIG.face_database_file):
    with open(path, "rb") as f:
        raw_database = pickle.load(f)

    normalized = []
    for item in raw_database:
        embedding = np.asarray(item["embedding"], dtype=np.float32)
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        employee_name = item.get("employee_name") or item.get("full_name")
        employee_code = item.get("employee_code")

        if employee_code is None and employee_name and "_" in employee_name:
            employee_code = employee_name.split("_", 1)[0]

        normalized.append({
            "employee_code": employee_code or employee_name,
            "employee_name": employee_name,
            "embedding": embedding,
        })

    return normalized


def save_face_database(face_database, path: Path = CONFIG.face_database_file):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(face_database, f)


def recognize_face(face_embedding, face_database, threshold: float = CONFIG.face_threshold):
    best_match = None
    best_score = -1.0
    embedding = np.asarray(face_embedding, dtype=np.float32)

    for data in face_database:
        score = float(np.dot(embedding, data["embedding"]))
        if score > best_score:
            best_score = score
            best_match = data

    if best_match is None or best_score < threshold:
        return {
            "employee_code": None,
            "employee_name": "Unknown",
            "score": best_score,
            "is_known": False,
        }

    return {
        "employee_code": best_match.get("employee_code"),
        "employee_name": best_match["employee_name"],
        "score": best_score,
        "is_known": True,
    }
