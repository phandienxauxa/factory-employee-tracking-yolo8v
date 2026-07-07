import argparse
import csv
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from config import CONFIG
from zone_loader import load_work_zones


ZONE_COLORS = [
    (255, 0, 0),
    (255, 0, 255),
    (255, 255, 0),
    (255, 255, 255),
    (128, 0, 128),
    (203, 192, 255),
    (255, 128, 0),
    (180, 180, 180),
]

ZONE_FONT_SCALE = 0.42
TRACK_FONT_SCALE = 0.42
TIMER_FONT_SCALE = 0.36
LOG_COLUMNS = ["thoi_gian", "track_id", "employee", "employee_code", "trang_thai", "so_giay_vang_mat"]


def parse_args():
    parser = argparse.ArgumentParser(description="YOLO zone-based work monitoring")
    parser.add_argument("--video", default=str(CONFIG.yolo_video_path), help="Path to input video")
    parser.add_argument("--model", default=str(CONFIG.yolo_model_path), help="YOLO/OpenVINO model path")
    parser.add_argument("--output-video", default=str(CONFIG.yolo_output_video_path), help="Path to save annotated tracking video")
    parser.add_argument("--no-save-video", action="store_true", help="Disable saving annotated tracking video")
    parser.add_argument("--no-display", action="store_true", help="Run without opening the OpenCV preview window")
    parser.add_argument("--log-file", default=str(CONFIG.yolo_log_file), help="CSV log path")
    parser.add_argument("--conf", type=float, default=CONFIG.yolo_confidence, help="Detection confidence threshold")
    parser.add_argument("--iou", type=float, default=CONFIG.yolo_iou, help="NMS IoU threshold")
    parser.add_argument("--imgsz", type=int, default=CONFIG.yolo_image_size, help="YOLO inference image size")
    parser.add_argument("--tracker", default=CONFIG.yolo_tracker, help="Ultralytics tracker config")
    parser.add_argument("--max-out-time", type=float, default=CONFIG.yolo_max_out_time)
    parser.add_argument(
        "--work-confirm-time",
        type=float,
        default=CONFIG.yolo_work_confirm_time,
        help="Seconds inside a work zone before a new person is counted as WORK",
    )
    parser.add_argument(
        "--return-confirm-time",
        type=float,
        default=CONFIG.yolo_return_confirm_time,
        help="Seconds inside a work zone before an AWAY person is counted as returned",
    )
    parser.add_argument("--frame-skip", type=int, default=CONFIG.yolo_frame_skip)
    return parser.parse_args()


def ensure_log_file(log_file):
    log_path = Path(log_file)
    if not log_path.exists():
        with open(log_path, mode="w", newline="", encoding="utf-8-sig") as file:
            writer = csv.writer(file)
            writer.writerow(LOG_COLUMNS)
        return

    with open(log_path, mode="r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None:
            rows = []
            current_columns = []
        else:
            current_columns = [column.strip() for column in reader.fieldnames]
            rows = list(reader)

    if current_columns == LOG_COLUMNS:
        return

    with open(log_path, mode="w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=LOG_COLUMNS)
        writer.writeheader()
        for row in rows:
            normalized_row = {key.strip(): value for key, value in row.items() if key is not None}
            writer.writerow({
                "thoi_gian": normalized_row.get("thoi_gian", ""),
                "track_id": normalized_row.get("track_id", ""),
                "employee": normalized_row.get("employee", normalized_row.get("employee_name", "")),
                "employee_code": normalized_row.get("employee_code", ""),
                "trang_thai": normalized_row.get("trang_thai", ""),
                "so_giay_vang_mat": normalized_row.get("so_giay_vang_mat", ""),
            })


def append_log(log_file, track_id, status, away_seconds="", assigned=None):
    employee = ""
    employee_code = ""
    if assigned is not None:
        employee = assigned.get("employee_name", "")
        employee_code = assigned.get("employee_code", "")

    with open(log_file, mode="a", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        writer.writerow([current_time, track_id, employee, employee_code, status, away_seconds])


def get_box_center(x1, y1, x2, y2):
    return int((x1 + x2) / 2), int((y1 + y2) / 2)


def point_in_polygon(point, polygon_points):
    polygon = np.array(polygon_points, dtype=np.int32)
    return cv2.pointPolygonTest(polygon, point, False) >= 0


def get_zone_for_person(x1, y1, x2, y2, work_zones):
    center = get_box_center(x1, y1, x2, y2)
    for zone_id, zone in work_zones.items():
        if point_in_polygon(center, zone["points"]):
            return zone_id, zone

    return None, None


def is_valid_person_box(x1, y1, x2, y2):
    box_area = (x2 - x1) * (y2 - y1)
    return box_area >= CONFIG.yolo_min_box_area


def draw_work_zones(frame, work_zones):
    for index, (zone_id, zone) in enumerate(work_zones.items()):
        points = np.array(zone["points"], dtype=np.int32)
        color = ZONE_COLORS[index % len(ZONE_COLORS)]

        cv2.polylines(frame, [points], isClosed=True, color=color, thickness=2)

        x, y = points[0]
        cv2.putText(
            frame,
            f"{zone_id}: {zone['zone_name']} - {zone['employee_code']}",
            (int(x), int(y) - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            ZONE_FONT_SCALE,
            color,
            1,
        )


def get_identity_label(track_id, assigned, zone):
    if assigned is not None:
        return f"ID {track_id} - {assigned['employee_code']} - {assigned['employee_name']}"

    if zone is not None:
        return f"ID {track_id} - waiting {zone['zone_name']}"

    return f"ID {track_id} - Unknown"


def assign_track_to_zone(
    track_id,
    zone_id,
    zone,
    current_time,
    track_zone_start_time,
    track_assigned_employee,
    zone_current_track,
):
    if track_id in track_assigned_employee:
        return

    if zone is None:
        track_zone_start_time.pop(track_id, None)
        return

    previous = track_zone_start_time.get(track_id)
    if previous is None or previous["zone_id"] != zone_id:
        track_zone_start_time[track_id] = {
            "zone_id": zone_id,
            "start_time": current_time,
        }
        return

    elapsed = (current_time - previous["start_time"]).total_seconds()
    if elapsed < CONFIG.yolo_zone_assign_seconds:
        return

    occupying_track = zone_current_track.get(zone_id)
    if occupying_track is not None and occupying_track != track_id:
        return

    track_assigned_employee[track_id] = {
        "employee_code": zone["employee_code"],
        "employee_name": zone["employee_name"],
        "zone_id": zone_id,
        "zone_name": zone["zone_name"],
    }
    zone_current_track[zone_id] = track_id


def release_stale_zone_assignments(track_last_seen, track_assigned_employee, zone_current_track):
    now = datetime.now()
    stale_seconds = max(CONFIG.yolo_max_out_time * 3, 10)

    for track_id, assigned in list(track_assigned_employee.items()):
        last_seen = track_last_seen.get(track_id)
        if last_seen is None:
            continue

        if (now - last_seen).total_seconds() <= stale_seconds:
            continue

        zone_id = assigned["zone_id"]
        if zone_current_track.get(zone_id) == track_id:
            del zone_current_track[zone_id]
        del track_assigned_employee[track_id]


def draw_ignored_person(frame, track_id, x1, y1, x2, y2):
    color = (160, 160, 160)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
    cv2.putText(
        frame,
        f"ID {track_id} - ignored",
        (x1, y1 - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        TRACK_FONT_SCALE,
        color,
        1,
    )


def create_video_writer(output_video, input_video, fps, frame_width, frame_height, frame_skip):
    if not output_video:
        return None

    output_path = Path(output_video)
    input_path = Path(input_video)

    try:
        if output_path.resolve() == input_path.resolve():
            print("Khong luu output video vi duong dan trung voi video input.")
            return None
    except OSError:
        pass

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_fps = max(fps / max(frame_skip, 1), 1)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, output_fps, (frame_width, frame_height))

    if not writer.isOpened():
        print(f"Khong tao duoc file video output: {output_path}")
        return None

    print(f"Se luu video ket qua vao: {output_path}")
    return writer


def main():
    args = parse_args()
    ensure_log_file(args.log_file)

    work_zones = load_work_zones(CONFIG.work_zones_file)
    if not work_zones:
        print("Chua co khu vuc lam viec nao trong work_zones.json.")
        print("Hay chay: python zone_editor.py")
        print("Trong cua so 'zone editor': click chuot trai de ve polygon, nhan Enter/n de chot zone.")
        print("Sau khi ve xong, chay lai: python yolo_tracking.py")
        return

    model = YOLO(args.model)
    cap = cv2.VideoCapture(args.video)

    if not cap.isOpened():
        print("Khong mo duoc video")
        return

    person_time = {}
    person_out_time = {}
    person_out_display_time = {}
    person_return_time = {}
    is_confirmed_working = {}
    has_logged_away = {}

    track_zone_start_time = {}
    track_assigned_employee = {}
    zone_current_track = {}
    track_last_seen = {}

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    time_per_processed_frame = max(args.frame_skip, 1) / fps
    frame_count = 0
    window_name = "yolo multi zone monitoring"
    video_writer = None

    if not args.no_save_video:
        video_writer = create_video_writer(
            args.output_video,
            args.video,
            fps,
            frame_width,
            frame_height,
            args.frame_skip,
        )

    print("Dang giam sat cac khu vuc, nhan q de thoat")
    if not args.no_display:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, CONFIG.display_window_width, CONFIG.display_window_height)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            if frame_count % max(args.frame_skip, 1) != 0:
                continue

            results = model.track(
                frame,
                persist=True,
                classes=0,
                conf=args.conf,
                iou=args.iou,
                imgsz=args.imgsz,
                tracker=args.tracker,
                verbose=False,
            )

            current_time = datetime.now()

            if results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
                ids = results[0].boxes.id.cpu().numpy().astype(int)

                for box, track_id in zip(boxes, ids):
                    x1, y1, x2, y2 = box
                    track_last_seen[track_id] = current_time

                    zone_id, zone = get_zone_for_person(x1, y1, x2, y2, work_zones)
                    is_inside_zone = zone is not None

                    if not is_valid_person_box(x1, y1, x2, y2) and not is_inside_zone:
                        draw_ignored_person(frame, track_id, x1, y1, x2, y2)
                        continue

                    assign_track_to_zone(
                        track_id,
                        zone_id,
                        zone,
                        current_time,
                        track_zone_start_time,
                        track_assigned_employee,
                        zone_current_track,
                    )

                    assigned = track_assigned_employee.get(track_id)
                    base_label = get_identity_label(track_id, assigned, zone)
                    status_text = "Unknown"
                    box_color = (0, 165, 255)

                    if is_inside_zone:
                        person_out_display_time[track_id] = 0.0

                        if has_logged_away.get(track_id, False):
                            person_return_time[track_id] = person_return_time.get(track_id, 0.0) + time_per_processed_frame

                            if person_return_time[track_id] >= args.return_confirm_time:
                                append_log(
                                    args.log_file,
                                    track_id,
                                    "da_quay_lai",
                                    round(person_out_time.get(track_id, 0.0), 1),
                                    assigned,
                                )
                                has_logged_away[track_id] = False
                                is_confirmed_working[track_id] = True
                                person_return_time[track_id] = 0.0
                                person_out_time[track_id] = 0.0
                                status_text = "RETURNED"
                                box_color = (0, 255, 0)
                            else:
                                status_text = "RETURNING"
                                box_color = (0, 255, 255)
                                cv2.putText(
                                    frame,
                                    f"return: {person_return_time[track_id]:.1f}/{args.return_confirm_time:.1f}s",
                                    (x1, y1 - 30),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    TIMER_FONT_SCALE,
                                    box_color,
                                    1,
                                )
                        elif assigned is not None:
                            is_confirmed_working[track_id] = True
                            person_out_time[track_id] = 0.0
                            person_return_time[track_id] = 0.0
                            person_time[track_id] = person_time.get(track_id, 0.0) + time_per_processed_frame
                            status_text = "WORK"
                            box_color = (0, 255, 0)
                        else:
                            waiting = track_zone_start_time.get(track_id)
                            wait_seconds = 0.0
                            if waiting is not None:
                                wait_seconds = (current_time - waiting["start_time"]).total_seconds()

                            status_text = "ENTERING"
                            box_color = (0, 255, 255)
                            cv2.putText(
                                frame,
                                f"wait: {wait_seconds:.1f}/{CONFIG.yolo_zone_assign_seconds:.1f}s",
                                (x1, y1 - 30),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                TIMER_FONT_SCALE,
                                box_color,
                                1,
                            )

                        cv2.putText(
                            frame,
                            f"work: {person_time.get(track_id, 0.0):.1f}s",
                            (x1, y1 - 15),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            TIMER_FONT_SCALE,
                            box_color,
                            1,
                        )
                    else:
                        track_zone_start_time.pop(track_id, None)
                        person_return_time[track_id] = 0.0
                        person_out_display_time[track_id] = person_out_display_time.get(track_id, 0.0) + time_per_processed_frame

                        if is_confirmed_working.get(track_id, False) or has_logged_away.get(track_id, False):
                            person_out_time[track_id] = person_out_time.get(track_id, 0.0) + time_per_processed_frame
                        else:
                            person_out_time[track_id] = 0.0

                        out_spent = person_out_time[track_id]
                        if has_logged_away.get(track_id, False) or (
                            is_confirmed_working.get(track_id, False) and out_spent > args.max_out_time
                        ):
                            if not has_logged_away.get(track_id, False):
                                append_log(args.log_file, track_id, "roi_khoi_vi_tri", assigned=assigned)
                                has_logged_away[track_id] = True
                                is_confirmed_working[track_id] = False

                            status_text = "AWAY"
                            box_color = (0, 0, 255)
                            cv2.putText(
                                frame,
                                f"WARNING: {out_spent:.1f}s",
                                (x1, y1 - 15),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                TIMER_FONT_SCALE,
                                box_color,
                                1,
                            )
                        else:
                            status_text = "OUT"
                            box_color = (0, 165, 255)
                            cv2.putText(
                                frame,
                                f"out: {person_out_display_time[track_id]:.1f}s",
                                (x1, y1 - 15),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                TIMER_FONT_SCALE,
                                box_color,
                                1,
                            )

                    label = f"{base_label} - {status_text}"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                    cv2.putText(
                        frame,
                        label,
                        (x1, y1 - 2),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        TRACK_FONT_SCALE,
                        box_color,
                        1,
                    )

            release_stale_zone_assignments(track_last_seen, track_assigned_employee, zone_current_track)
            draw_work_zones(frame, work_zones)

            if video_writer is not None:
                video_writer.write(frame)

            if not args.no_display:
                cv2.imshow(window_name, frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        cap.release()
        if video_writer is not None:
            video_writer.release()
            print(f"Da luu video ket qua: {args.output_video}")
        if not args.no_display:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
