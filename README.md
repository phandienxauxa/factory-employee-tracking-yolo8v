# Camera Attendance & Work-Zone Monitoring

Hệ thống demo chấm công và giám sát khu vực làm việc bằng camera, kết hợp nhận diện khuôn mặt, lưu log SQLite, dashboard Streamlit và theo dõi nhân sự theo vùng bằng YOLO/OpenVINO.

## Tổng Quan

Dự án có 2 luồng xử lý chính:

- **Face Attendance**: nhận diện nhân viên từ webcam bằng InsightFace, ghi sự kiện `CHECK_IN` / `CHECK_OUT` vào SQLite và xem lại trên dashboard.
- **Work-Zone Monitoring**: theo dõi người trong các vùng làm việc đã vẽ trước, xác định trạng thái `WORK`, `AWAY`, `RETURNING`, `RETURNED` và ghi lịch sử ra CSV.

Repo này tập trung vào demo chạy local, phù hợp để nghiên cứu pipeline computer vision trước khi đóng gói thành hệ thống production.

## Tính Năng Chính

- Nhận diện khuôn mặt nhân viên từ webcam.
- Tạo database embedding khuôn mặt từ thư mục dataset.
- Lưu lịch sử chấm công bằng SQLite.
- Dashboard Streamlit để xem dữ liệu chấm công.
- Tracking người bằng YOLO với Ultralytics + ByteTrack.
- Hỗ trợ model YOLO export sang OpenVINO.
- Vẽ và lưu vùng làm việc bằng công cụ OpenCV `zone_editor.py`.
- Ghi log nhân sự rời vị trí và quay lại vào `history.csv`.

## Công Nghệ

- Python 3.10+
- OpenCV
- InsightFace
- ONNX Runtime
- SQLite
- Streamlit
- Ultralytics YOLO
- OpenVINO
- NumPy / Pandas

## Kiến Trúc Dự Án

```text
camera_attendance_demo/
├── config.py                         # Cấu hình đường dẫn, threshold, timing
├── init_database.py                  # Tạo / migrate SQLite database
├── database_service.py               # Service thao tác dữ liệu chấm công
├── face_service.py                   # Load embedding và nhận diện khuôn mặt
├── create_embeddings.py              # Tạo face_database.pkl từ dataset/
├── attendance_webcam_db.py           # Entry point chấm công webcam
├── attendance_webcam_db_checkout_test.py
│                                      # Logic nhận diện và check-in/check-out
├── dashboard_db.py                   # Dashboard Streamlit cho SQLite
├── yolo_tracking.py                  # Theo dõi người trong vùng làm việc
├── zone_editor.py                    # Vẽ vùng làm việc trên video
├── zone_loader.py                    # Load work_zones.json
├── work_zones.py                     # Vùng mẫu dạng Python dict
├── extract_frame.py                  # Cắt frame từ video để tạo dataset train
├── export_openvino.py                # Export YOLO .pt sang OpenVINO
├── requirements.txt
└── README.md
```

## Dữ Liệu Và Model Không Commit

Các file/thư mục sau được ignore để repo nhẹ và an toàn hơn:

```text
venv/
dataset/
captures/
database.db
face_database.pkl
attendance_logs.csv
history.csv
*.mp4
*.pt
*_openvino_model/
```

Khi clone repo mới, bạn cần tự chuẩn bị lại:

- `dataset/` chứa ảnh khuôn mặt nhân viên.
- `demo.mp4` hoặc video/camera input cho YOLO tracking.
- `best.pt` hoặc thư mục model OpenVINO, ví dụ `best_openvino_model/`.
- `face_database.pkl`, tạo bằng `create_embeddings.py`.
- `database.db`, tạo bằng `init_database.py`.

## Cài Đặt

Tạo môi trường ảo:

```powershell
python -m venv venv
.\venv\Scripts\activate
```

Cài dependency:

```powershell
pip install -r requirements.txt
```

Nếu dùng GPU hoặc môi trường đặc biệt, bạn có thể cần cài bản `onnxruntime`, OpenVINO hoặc Ultralytics phù hợp với máy.

## Cấu Hình

Cấu hình chính nằm trong `config.py`:

```python
dataset_dir = BASE_DIR / "dataset"
face_database_file = BASE_DIR / "face_database.pkl"
database_file = BASE_DIR / "database.db"
yolo_model_path = BASE_DIR / "best_openvino_model"
yolo_video_path = BASE_DIR / "demo.mp4"
yolo_image_size = 640
yolo_confidence = 0.35
yolo_frame_skip = 1
```

Model OpenVINO hiện được cấu hình chạy với input size `640`. Nếu model export có input cố định, không nên tăng `--imgsz` lên `960` hoặc `1280` khi chạy inference.

## Chuẩn Bị Dataset Khuôn Mặt

Tổ chức ảnh nhân viên theo format:

```text
dataset/<ma_nv>_<ten_nv>/
```

Ví dụ:

```text
dataset/NV001_NgoDangTao/
dataset/NV002_ThaiDucThien/
```

Sau đó tạo embedding:

```powershell
python create_embeddings.py
```

Lệnh này tạo file:

```text
face_database.pkl
```

## Chạy Chấm Công Bằng Webcam

Tạo hoặc cập nhật database:

```powershell
python init_database.py
```

Tạo embedding khuôn mặt nếu chưa có:

```powershell
python create_embeddings.py
```

Chạy nhận diện và ghi chấm công:

```powershell
python attendance_webcam_db.py
```

Mở dashboard:

```powershell
streamlit run dashboard_db.py
```

## Chạy Work-Zone Monitoring

### 1. Vẽ vùng làm việc

Chạy công cụ vẽ zone:

```powershell
python zone_editor.py
```

Phím điều khiển:

```text
Left click   Thêm điểm polygon
Right click  Xóa zone cuối
Enter / n    Lưu zone hiện tại
z / u        Undo điểm đang vẽ
c            Xóa toàn bộ zone
r            Reset polygon đang vẽ
s            Lưu work_zones.json
Space / p    Pause / resume video
a / d        Lùi / tiến frame khi pause
q            Thoát
```

Kết quả được lưu vào:

```text
work_zones.json
```

### 2. Chạy tracking

Chạy mặc định theo `config.py`:

```powershell
python yolo_tracking.py
```

Chạy với video/model chỉ định:

```powershell
python yolo_tracking.py --video demo.mp4 --model best_openvino_model
```

Lệnh gợi ý cho model OpenVINO input size 640:

```powershell
python yolo_tracking.py --video demo.mp4 --model best_openvino_model --conf 0.35 --imgsz 640 --frame-skip 1
```

Nếu cần chỉ định tracker:

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

## Tham Số YOLO Quan Trọng

| Tham số | Mặc định | Mô tả |
| --- | ---: | --- |
| `--conf` | `0.35` | Ngưỡng confidence phát hiện người. Giảm nếu người ở xa bị mất box. |
| `--iou` | `0.5` | Ngưỡng IoU cho NMS. |
| `--imgsz` | `640` | Kích thước input inference. Giữ `640` với model OpenVINO hiện tại. |
| `--frame-skip` | `1` | Số frame bỏ qua giữa các lần inference. |
| `--max-out-time` | `5.0` | Số giây được phép rời vùng trước khi bị tính `AWAY`. |
| `--work-confirm-time` | `3.0` | Số giây cần ở trong vùng để được xác nhận `WORK`. |
| `--return-confirm-time` | `3.0` | Số giây cần ở lại vùng sau khi quay lại để ghi `RETURNED`. |

## File Log

SQLite chấm công:

```text
database.db
```

Log tracking khu vực:

```text
history.csv
```

Các cột trong `history.csv`:

```text
thoi_gian,track_id,employee,employee_code,trang_thai,so_giay_vang_mat
```

Sự kiện có thể xuất hiện:

- `roi_khoi_vi_tri`
- `da_quay_lai`

Reset log YOLO nhưng giữ header:

```powershell
Set-Content -Path history.csv -Value "thoi_gian,track_id,employee,employee_code,trang_thai,so_giay_vang_mat" -Encoding UTF8
```

## Train Và Export YOLO

Pipeline đã dùng trong demo:

1. Cắt frame từ video bằng `extract_frame.py`.
2. Gán nhãn / quản lý dataset trên Roboflow.
3. Train YOLO trên Google Colab.
4. Tải model `.pt` về local.
5. Export sang OpenVINO:

```powershell
python export_openvino.py
```

Thư mục model OpenVINO nên đặt theo cấu hình:

```text
best_openvino_model/
```

## Kiểm Tra Nhanh Trước Khi Push

Kiểm tra compile Python:

```powershell
python -m py_compile config.py work_zones.py zone_loader.py zone_editor.py yolo_tracking.py
```

Kiểm tra file sẽ được commit:

```powershell
git status --short
git check-ignore -v database.db face_database.pkl demo.mp4 best.pt best_openvino_model venv dataset captures
```

Không commit các file dữ liệu thật, ảnh nhân viên, database runtime, video demo hoặc model lớn lên GitHub public.

## Troubleshooting

### Không mở được webcam

- Kiểm tra `camera_index` trong `config.py`.
- Đóng các ứng dụng khác đang dùng camera.
- Thử đổi `camera_index` từ `0` sang `1`.

### InsightFace không nhận diện đúng

- Tăng số ảnh và độ đa dạng góc mặt trong `dataset/`.
- Chạy lại `python create_embeddings.py`.
- Kiểm tra `face_threshold` trong `config.py`.

### YOLO bị mất người ở xa

- Giảm nhẹ `--conf`, ví dụ `0.25` đến `0.35`.
- Giữ `--frame-skip 1` để tracking ổn định hơn.
- Bổ sung dữ liệu train ở góc camera xa.
- Không tăng `--imgsz` nếu model OpenVINO export fixed-size `640`.

### `work_zones.json` sai vùng

- Chạy lại `python zone_editor.py`.
- Vẽ zone trên cùng video/cùng độ phân giải với video tracking.
- Kiểm tra lại mapping `employee_code` / `employee_name` trong `zone_editor.py`.

## Roadmap Đề Xuất

- Đổi tên `attendance_webcam_db_checkout_test.py` thành entry module chính thức hơn.
- Tách source code vào package `src/`.
- Thêm CLI thống nhất cho face attendance và YOLO tracking.
- Thêm migration version cho SQLite schema.
- Thêm test unit cho `zone_loader.py`, `database_service.py` và state machine trong `yolo_tracking.py`.
- Bổ sung Dockerfile hoặc script setup môi trường.

## License

Chưa khai báo license. Nếu muốn public repo, nên thêm file `LICENSE` trước khi chia sẻ rộng rãi.
