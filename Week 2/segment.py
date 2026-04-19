from ultralytics import YOLO

# Segmentation model (downloads ~7MB automatically)
model = YOLO("yolo11n-seg.pt")

# Run segmentation on bus.jpg with save=True
results = model("bus.jpg", save=True)

for result in results:
    print(result)
    print("\n✅ Segmentation complete!")
    print("Masks found:", result.masks is not None)
    print("Save location:", result.save_dir)