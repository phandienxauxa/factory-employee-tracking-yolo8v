# Factory Employee Tracking YOLOv8

Ứng dụng demo giám sát nhân sự trong khu vực làm việc bằng camera/video, sử dụng YOLOv8, ByteTrack và OpenVINO. Dự án tập trung vào bài toán phát hiện người, tracking theo vùng làm việc và ghi nhận trạng thái rời vị trí/quay lại.

## Demo Video

Video demo được commit trực tiếp trong repo để người xem có thể mở ngay trên GitHub.

### Before Tracking

Video gốc trước khi chạy hệ thống tracking:

<video src="./demo.mp4" controls width="720"></video>

[Mở demo.mp4](./demo.mp4)

### After Tracking

Chạy tracking và lưu video kết quả ra file:

```powershell
python yolo_tracking.py --video demo.mp4 --model best_openvino_model --output-video demo_after_tracking.mp4
```

Nếu chỉ muốn render video, không mở cửa sổ OpenCV:

```powershell
python yolo_tracking.py --video demo.mp4 --model best_openvino_model --output-video demo_after_tracking.mp4 --no-display
```

Video kết quả:

[Mở demo_after_tracking.mp4](./demo_after_tracking.mp4)

## Tính Năng

- Phát hiện người bằng YOLOv8.
- Tracking ID bằng Ultralytics tracker/ByteTrack.
- Vẽ vùng làm việc bằng công cụ OpenCV `zone_editor.py`.
- Load vùng làm việc từ `work_zones.json`.
- Gán mỗi vùng với một nhân sự/khu vực cụ thể.
- Theo dõi trạng thái `OUT`, `ENTERING`, `WORK`, `AWAY`, `RETURNING`, `RETURNED`.
- Ghi log sự kiện rời vị trí và quay lại vào `history.csv`.
- Lưu video tracking đã vẽ bounding box, zone và trạng thái ra `demo_after_tracking.mp4`.
- Hỗ trợ chạy model YOLO export sang OpenVINO để tối ưu inference local.

## Công Nghệ

- Python 3.10+
- OpenCV
- NumPy
- Ultralytics YOLO
- ByteTrack
- OpenVINO

## Cấu Trúc Dự Án

```text
factory-employee-tracking-yolo8v/
├── config.py                 # Cấu hình model, video, threshold, thời gian trạng thái
├── yolo_tracking.py          # Entry point tracking nhân sự theo vùng làm việc
├── zone_editor.py            # Công cụ vẽ vùng làm việc trên video
├── zone_loader.py            # Load và validate work_zones.json
├── work_zones.json           # Vùng làm việc đã vẽ
├── work_zones.py             # Vùng mẫu dạng Python dict
├── extract_frame.py          # Cắt frame từ video để tạo dữ liệu train
├── export_openvino.py        # Export YOLO .pt sang OpenVINO
├── demo.mp4                  # Video demo gốc
├── demo_after_tracking.mp4   # Video kết quả sau tracking, tạo bằng --output-video
├── requirements.txt
└── README.md
```

## File Không Commit

Repo vẫn bỏ qua các file runtime hoặc file nặng không cần thiết:

```text
venv/
__pycache__/
history.csv
captures/
frames/
*.pt
*_openvino_model/
```

`demo.mp4` và `demo_after_tracking.mp4` không bị ignore để có thể hiển thị trực tiếp trên GitHub.

## Cài Đặt

Tạo môi trường ảo:

```powershell
python -m venv venv
.\venv\Scripts\activate
```

Cài dependencies:

```powershell
pip install -r requirements.txt
```

## Cấu Hình

Cấu hình chính nằm trong `config.py`:

```python
yolo_model_path = BASE_DIR / "best_openvino_model"
yolo_video_path = BASE_DIR / "demo.mp4"
yolo_log_file = BASE_DIR / "history.csv"
yolo_confidence = 0.35
yolo_iou = 0.5
yolo_image_size = 640
yolo_frame_skip = 1
```

Model OpenVINO hiện được cấu hình với input size `640`. Nếu model export có input cố định, nên giữ `--imgsz 640`.

## Vẽ Vùng Làm Việc

Chạy:

```powershell
python zone_editor.py
```

Điều khiển:

| Phím / thao tác | Chức năng |
| --- | --- |
| Left click | Thêm điểm polygon |
| Right click | Xóa zone cuối |
| Enter / n | Lưu zone hiện tại |
| z / u | Undo điểm đang vẽ |
| c | Xóa toàn bộ zone |
| r | Reset polygon đang vẽ |
| s | Lưu `work_zones.json` |
| Space / p | Pause / resume video |
| a / d | Lùi / tiến frame khi pause |
| q | Thoát |

Kết quả được lưu vào:

```text
work_zones.json
```

## Chạy Tracking

Chạy theo cấu hình mặc định:

```powershell
python yolo_tracking.py
```

Mặc định chương trình lưu video đã annotate vào:

```text
demo_after_tracking.mp4
```

Chỉ định video và model:

```powershell
python yolo_tracking.py --video demo.mp4 --model best_openvino_model
```

Lệnh gợi ý cho model OpenVINO input size 640:

```powershell
python yolo_tracking.py --video demo.mp4 --model best_openvino_model --conf 0.35 --imgsz 640 --frame-skip 1
```

Chỉ định nơi lưu video output:

```powershell
python yolo_tracking.py --video demo.mp4 --model best_openvino_model --output-video demo_after_tracking.mp4
```

Tắt lưu video:

```powershell
python yolo_tracking.py --no-save-video
```

Render video không mở cửa sổ preview:

```powershell
python yolo_tracking.py --output-video demo_after_tracking.mp4 --no-display
```

Chỉ định tracker:

```powershell
python yolo_tracking.py --tracker ".\venv\Lib\site-packages\ultralytics\cfg\trackers\bytetrack.yaml"
```

## Trạng Thái Tracking

| Trạng thái | Ý nghĩa |
| --- | --- |
| `OUT` | Người đang ở ngoài vùng làm việc. Track mới xuất hiện ngoài vùng chưa bị tính vi phạm. |
| `ENTERING` | Người mới đi vào vùng, đang chờ đủ thời gian xác nhận `WORK`. |
| `WORK` | Người đã ở trong vùng đủ `--work-confirm-time` giây. |
| `AWAY` | Người từng được xác nhận `WORK`, sau đó rời vùng quá `--max-out-time` giây. |
| `RETURNING` | Người đang quay lại vùng sau khi bị `AWAY`, chưa đủ thời gian xác nhận. |
| `RETURNED` | Người đã quay lại đủ `--return-confirm-time` giây, hệ thống ghi log `da_quay_lai`. |

## Tham Số Quan Trọng

| Tham số | Mặc định | Mô tả |
| --- | ---: | --- |
| `--conf` | `0.35` | Ngưỡng confidence phát hiện người. |
| `--iou` | `0.5` | Ngưỡng IoU cho NMS. |
| `--imgsz` | `640` | Kích thước input inference. |
| `--frame-skip` | `1` | Số frame bỏ qua giữa các lần inference. |
| `--max-out-time` | `5.0` | Số giây được phép rời vùng trước khi bị tính `AWAY`. |
| `--work-confirm-time` | `3.0` | Số giây cần ở trong vùng để được xác nhận `WORK`. |
| `--return-confirm-time` | `3.0` | Số giây cần ở lại vùng sau khi quay lại để ghi `RETURNED`. |

## Log Kết Quả

Tracking ghi sự kiện vào:

```text
history.csv
```

Các cột:

```text
thoi_gian,track_id,employee,employee_code,trang_thai,so_giay_vang_mat
```

Sự kiện chính:

- `roi_khoi_vi_tri`
- `da_quay_lai`

Reset log nhưng giữ header:

```powershell
Set-Content -Path history.csv -Value "thoi_gian,track_id,employee,employee_code,trang_thai,so_giay_vang_mat" -Encoding UTF8
```

## Train Và Export Model

Pipeline đề xuất:

1. Cắt frame từ video bằng `extract_frame.py`.
2. Gán nhãn người trong ảnh.
3. Train YOLOv8.
4. Tải model `.pt` về local.
5. Export sang OpenVINO:

```powershell
python export_openvino.py
```

Thư mục model OpenVINO mặc định:

```text
best_openvino_model/
```

## Kiểm Tra Trước Khi Push

Compile nhanh:

```powershell
python -m py_compile config.py work_zones.py zone_loader.py zone_editor.py yolo_tracking.py extract_frame.py export_openvino.py
```

Kiểm tra trạng thái Git:

```powershell
git status --short
```

## Troubleshooting

### Không mở được video

- Kiểm tra `demo.mp4` có tồn tại trong thư mục repo.
- Kiểm tra `yolo_video_path` trong `config.py`.
- Thử truyền video trực tiếp qua `--video`.

### Không load được model

- Kiểm tra `best_openvino_model/` hoặc đường dẫn truyền vào `--model`.
- Nếu dùng `.pt`, truyền trực tiếp file `.pt` qua `--model`.
- Đảm bảo phiên bản `ultralytics` và `openvino` đã được cài đúng.

### Tracking bị mất người ở xa

- Giảm nhẹ `--conf`, ví dụ `0.25` đến `0.35`.
- Giữ `--frame-skip 1` để tracking ổn định hơn.
- Bổ sung dữ liệu train ở góc camera xa.
- Không tăng `--imgsz` nếu model OpenVINO export fixed-size `640`.

### Vùng làm việc bị lệch

- Chạy lại `python zone_editor.py`.
- Vẽ zone trên cùng video/cùng độ phân giải với video tracking.
- Kiểm tra mapping `employee_code` / `employee_name` trong `zone_editor.py`.

## Roadmap

- Tách state machine tracking thành module riêng để dễ test.
- Thêm unit test cho `zone_loader.py`.
- Thêm Dockerfile hoặc script setup môi trường.

## License

Xem file `LICENSE`.
