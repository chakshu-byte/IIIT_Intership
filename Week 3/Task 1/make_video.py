import cv2

# Paths to 3 versions of the image
raw_img      = cv2.imread("bus.jpg")
detected_img = cv2.imread("runs/detect/predict/bus.jpg")
segmented_img = cv2.imread("runs/segment/predict/bus.jpg")

# Resize all to same size
h, w = 480, 640
raw_img       = cv2.resize(raw_img,       (w, h))
detected_img  = cv2.resize(detected_img,  (w, h))
segmented_img = cv2.resize(segmented_img, (w, h))

# Stack vertically
stacked = cv2.vconcat([raw_img, detected_img, segmented_img])

# Write to video (each frame shown for 3 seconds at 30fps = 90 frames)
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter("stacked_video.mp4", fourcc, 30, (w, h * 3))

for _ in range(90):  # 3 seconds
    out.write(stacked)

out.release()
print("✅ Stacked video saved as stacked_video.mp4")
print(f"Dimensions: {w}x{h*3} (3 panels of {w}x{h})")