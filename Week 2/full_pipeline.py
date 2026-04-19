from ultralytics import YOLO
import cv2, os

# Load models
detect_model = YOLO("yolo11n.pt")
seg_model    = YOLO("yolo11n-seg.pt")

# Extract frames from raw video
os.makedirs("frames_raw", exist_ok=True)
os.makedirs("frames_detected", exist_ok=True)
os.makedirs("frames_segmented", exist_ok=True)

cap = cv2.VideoCapture("raw.mp4")
fps = cap.get(cv2.CAP_PROP_FPS)
w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

i = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    cv2.imwrite(f"frames_raw/frame_{i:04d}.jpg", frame)
    i += 1
cap.release()
print(f"✅ Extracted {i} frames")

# Run detection + segmentation on each frame
for j in range(i):
    path = f"frames_raw/frame_{j:04d}.jpg"

    det = detect_model(path)
    det[0].save(filename=f"frames_detected/frame_{j:04d}.jpg")

    seg = seg_model(path)
    seg[0].save(filename=f"frames_segmented/frame_{j:04d}.jpg")

    if j % 10 == 0:
        print(f"Processed {j}/{i} frames...")

print("✅ Detection + Segmentation done!")

# Stitch 3 videos side by side (vstack)
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter("stacked_moving.mp4", fourcc, fps, (w, h*3))

for j in range(i):
    raw  = cv2.resize(cv2.imread(f"frames_raw/frame_{j:04d}.jpg"),       (w, h))
    det  = cv2.resize(cv2.imread(f"frames_detected/frame_{j:04d}.jpg"),  (w, h))
    seg  = cv2.resize(cv2.imread(f"frames_segmented/frame_{j:04d}.jpg"), (w, h))
    out.write(cv2.vconcat([raw, det, seg]))

out.release()
print("✅ Stacked video saved as stacked_moving.mp4")