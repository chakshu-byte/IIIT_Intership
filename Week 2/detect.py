from ultralytics import YOLO

# Load pretrained YOLO model (downloads ~6MB automatically)
model = YOLO("yolo11n.pt")

# Run detection on sample image
results = model("https://ultralytics.com/images/bus.jpg")

# Save and show results
for result in results:
    result.save(filename="output_detected.jpg")
    print(result)