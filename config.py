from dataclasses import dataclass
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class AppConfig:
    yolo_model_path: Path = BASE_DIR / "best_openvino_model"
    yolo_video_path: Path = BASE_DIR / "demo.mp4"
    yolo_log_file: Path = BASE_DIR / "history.csv"
    yolo_max_out_time: float = 5.0
    yolo_work_confirm_time: float = 3.0
    yolo_return_confirm_time: float = 3.0
    yolo_frame_skip: int = 1
    yolo_confidence: float = 0.35
    yolo_iou: float = 0.5
    yolo_image_size: int = 640
    yolo_tracker: str = str(BASE_DIR / "venv" / "Lib" / "site-packages" / "ultralytics" / "cfg" / "trackers" / "bytetrack.yaml")

    yolo_min_box_area: int = 5000
    yolo_zone_assign_seconds: float = 3.0
    work_zones_file: Path = BASE_DIR / "work_zones.json"
    display_window_width: int = 1280
    display_window_height: int = 720


CONFIG = AppConfig()
