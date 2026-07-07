import cv2

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Khong mo duoc webcam")
    exit()

while True:
    ret, frame = cap.read()

    if not ret:
        print("Khong doc duoc hinh anh tu webcam")
        break

    cv2.imshow("Webcam test", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()