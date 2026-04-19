from ultralytics import YOLO

model = YOLO("yolo11n.pt")
results = model("bus.jpg", save=True)

for result in results:
    print(result)