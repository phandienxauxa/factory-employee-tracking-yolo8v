import json
from pathlib import Path

import cv2
import numpy as np

from config import CONFIG


WORKER_ASSIGNMENTS = [
    {
        "zone_name": "Ban 001",
        "employee_code": "NV001",
        "employee_name": "DucThien",
    },
    {
        "zone_name": "Ban 002",
        "employee_code": "NV002",
        "employee_name": "DangTao",
    },
    {
        "zone_name": "Ban 003",
        "employee_code": "NV003",
        "employee_name": "ThanhSon",
    },
]


WINDOW_NAME = "zone editor"


def get_assignment(zone_index):
    if zone_index < len(WORKER_ASSIGNMENTS):
        return WORKER_ASSIGNMENTS[zone_index]

    zone_number = zone_index + 1
    return {
        "zone_name": f"Zone {zone_number}",
        "employee_code": "UNASSIGNED",
        "employee_name": "Unassigned",
    }


def draw_text(frame, text, y, color=(0, 255, 255)):
    cv2.putText(frame, text, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1)


def draw_saved_zones(frame, zones):
    for zone in zones:
        points = np.array(zone["points"], dtype=np.int32)
        cv2.polylines(frame, [points], isClosed=True, color=(255, 0, 0), thickness=2)

        x, y = points[0]
        cv2.putText(
            frame,
            f"{zone['zone_id']}: {zone['employee_code']} - {zone['employee_name']}",
            (int(x), int(y) - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (255, 0, 0),
            1,
        )


def draw_current_polygon(frame, current_points):
    if not current_points:
        return

    for point in current_points:
        cv2.circle(frame, point, 4, (0, 255, 255), -1)

    if len(current_points) >= 2:
        cv2.polylines(frame, [np.array(current_points, dtype=np.int32)], False, (0, 255, 255), 2)

    if len(current_points) >= 3:
        cv2.polylines(frame, [np.array(current_points, dtype=np.int32)], True, (0, 255, 255), 1)


def save_zones(zones, output_path):
    output = {"zones": zones}
    path = Path(output_path)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(output, file, ensure_ascii=False, indent=2)

    print(f"Da luu {len(zones)} zone vao: {path}")


def make_zone(zone_index, points):
    assignment = get_assignment(zone_index)
    zone_number = zone_index + 1
    return {
        "zone_id": f"zone_{zone_number}",
        "zone_name": assignment["zone_name"],
        "employee_code": assignment["employee_code"],
        "employee_name": assignment["employee_name"],
        "points": [[int(x), int(y)] for x, y in points],
    }


def add_current_zone(zones, current_points):
    if len(current_points) < 3:
        print("Polygon can it nhat 3 diem. Khong luu zone.")
        return False

    zone = make_zone(len(zones), current_points)
    zones.append(zone)
    print(f"Da them {zone['zone_id']}: {zone['employee_code']} - {zone['employee_name']}")
    current_points.clear()
    return True


def remove_last_zone(zones):
    if not zones:
        print("Chua co zone nao de xoa.")
        return False

    zone = zones.pop()
    print(f"Da xoa {zone['zone_id']}: {zone['employee_code']} - {zone['employee_name']}")
    return True


def clear_all_zones(zones, current_points):
    zones.clear()
    current_points.clear()
    print("Da xoa toan bo khu vuc lam viec.")


def main():
    video_path = str(CONFIG.yolo_video_path)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Khong mo duoc video: {video_path}")
        return

    zones = []
    current_points = []
    save_zones(zones, CONFIG.work_zones_file)
    print("Da bat dau phien ve moi. work_zones.json hien dang rong.")

    paused = True
    frame_index = 0
    ret, frame = cap.read()
    if not ret:
        print("Video bi loi hoac khong co frame.")
        cap.release()
        return

    original_height, original_width = frame.shape[:2]
    display_width = int(CONFIG.display_window_width)
    display_height = int(CONFIG.display_window_height)
    scale_x = original_width / display_width
    scale_y = original_height / display_height

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            original_x = max(0, min(original_width - 1, int(x * scale_x)))
            original_y = max(0, min(original_height - 1, int(y * scale_y)))
            current_points.append((original_x, original_y))
            print(f"Da them diem: ({original_x}, {original_y})")
        elif event == cv2.EVENT_RBUTTONDOWN:
            if remove_last_zone(zones):
                save_zones(zones, CONFIG.work_zones_file)

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, CONFIG.display_window_width, CONFIG.display_window_height)
    cv2.setMouseCallback(WINDOW_NAME, mouse_callback)

    print("Left click: add point | Right click: remove last zone | Enter/n: next zone | z/u: undo | c: clear all | r: reset current | s: save | q: quit")

    while True:
        display = frame.copy()
        next_assignment = get_assignment(len(zones))

        draw_saved_zones(display, zones)
        draw_current_polygon(display, current_points)
        draw_text(display, "Left: add point | Right: delete last zone | Enter/n: save zone | z/u: undo | c: clear all | q: quit", 24)
        draw_text(
            display,
            f"Drawing zone_{len(zones) + 1}: {next_assignment['zone_name']} - {next_assignment['employee_code']} - {next_assignment['employee_name']}",
            44,
        )
        draw_text(display, f"Frame: {frame_index} | Paused: {paused}", 64, (255, 255, 255))

        display = cv2.resize(display, (display_width, display_height))
        cv2.imshow(WINDOW_NAME, display)
        key = cv2.waitKey(30) & 0xFF

        if key == ord("q"):
            break
        if key == ord(" ") or key == ord("p"):
            paused = not paused
        elif key == ord("z") or key == ord("u"):
            if current_points:
                current_points.pop()
        elif key == ord("c"):
            clear_all_zones(zones, current_points)
            save_zones(zones, CONFIG.work_zones_file)
        elif key == ord("r"):
            current_points.clear()
        elif key == ord("n") or key == 13:
            if add_current_zone(zones, current_points):
                save_zones(zones, CONFIG.work_zones_file)
        elif key == ord("s"):
            save_zones(zones, CONFIG.work_zones_file)
        elif key == ord("d") and paused:
            ret, next_frame = cap.read()
            if ret:
                frame = next_frame
                frame_index += 1
        elif key == ord("a") and paused:
            frame_index = max(frame_index - 1, 0)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ret, prev_frame = cap.read()
            if ret:
                frame = prev_frame

        if not paused:
            ret, next_frame = cap.read()
            if ret:
                frame = next_frame
                frame_index += 1
            else:
                paused = True

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

