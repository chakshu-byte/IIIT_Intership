from PIL import Image
import os

FOLDERS = [
    r"C:\Users\apps\Downloads\Week4\Dataset\images\train",
    r"C:\Users\apps\Downloads\Week4\Dataset\images\val",
    r"C:\Users\apps\Downloads\Week4\Dataset\images\test",
]

TARGET_WIDTH = 384

for folder in FOLDERS:
    files = [f for f in os.listdir(folder) if f.endswith(".png")]
    print(f"\nProcessing {folder} — {len(files)} images")
    for fname in files:
        path = os.path.join(folder, fname)
        img = Image.open(path)
        w, h = img.size
        new_h = int(h * TARGET_WIDTH / w)
        img_resized = img.resize((TARGET_WIDTH, new_h), Image.LANCZOS)
        img_resized.save(path)
    print(f"  ✅ Done — resized to {TARGET_WIDTH}xH (aspect preserved)")

print("\n✅ All images scaled successfully!")