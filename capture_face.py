import cv2
import os

employee_id = "NV002"
employee_name = "ThaiDucThien"

save_dir = f"dataset/{employee_id}_{employee_name}"
os.makedirs(save_dir, exist_ok=True)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Khong mo duoc webcam")
    exit()

count = 0

print("Nhan S de luu anh")
print("Nhan Q de thoat")

while True:
    ret, frame = cap.read()

    if not ret:
        print("Khong doc duoc hinh anh tu webcam")
        break

    cv2.putText(
        frame,
        "Nhan S de luu anh, Q de thoat",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.imshow("Capture full frame", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("s"):
        count += 1
        filename = f"{save_dir}/{employee_id}_{count}.jpg"

        cv2.imwrite(filename, frame)
        print(f"Da luu anh: {filename}")

    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()