import json
from pathlib import Path


def _valid_points(points):
    if not isinstance(points, list) or len(points) < 3:
        return False

    for point in points:
        if not isinstance(point, list) and not isinstance(point, tuple):
            return False
        if len(point) != 2:
            return False
        if not all(isinstance(value, (int, float)) for value in point):
            return False

    return True


def load_work_zones(json_path):
    path = Path(json_path)
    if not path.exists():
        print("Chua co work_zones.json. Hay chay zone_editor.py de ve khu vuc lam viec.")
        return {}

    try:
        with open(path, "r", encoding="utf-8-sig") as file:
            data = json.load(file)
    except json.JSONDecodeError as exc:
        print(f"File work_zones.json loi format: {exc}")
        return {}
    except OSError as exc:
        print(f"Khong doc duoc file work_zones.json: {exc}")
        return {}

    zones = data.get("zones", [])
    if not isinstance(zones, list):
        print("File work_zones.json khong hop le: 'zones' phai la list.")
        return {}

    work_zones = {}
    for index, zone in enumerate(zones, start=1):
        if not isinstance(zone, dict):
            print(f"Bo qua zone thu {index}: du lieu khong phai object.")
            continue

        points = zone.get("points")
        if not _valid_points(points):
            print(f"Bo qua {zone.get('zone_id', f'zone_{index}')}: points khong hop le.")
            continue

        zone_id = zone.get("zone_id") or f"zone_{index}"
        work_zones[zone_id] = {
            "zone_name": zone.get("zone_name", f"Zone {index}"),
            "employee_code": zone.get("employee_code", "UNASSIGNED"),
            "employee_name": zone.get("employee_name", "Unassigned"),
            "points": [(int(x), int(y)) for x, y in points],
        }

    return work_zones
