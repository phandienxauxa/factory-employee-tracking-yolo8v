import cv2 
import os 

video_path = "demo.mp4"
save_dir = "frames"
frame_interval = 10 

os.makedirs(save_dir, exist_ok = True)

cap = cv2.VideoCapture(video_path)

frame_id = 0
saved_id = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    if frame_id % frame_interval == 0:
        filename = os.path.join(save_dir, f"frame_{saved_id:05d}.jpg")
        cv2.imwrite(filename, frame)
        saved_id += 1
    frame_id += 1

cap.release()

print("Done, Total frames saved:", saved_id)