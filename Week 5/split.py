import os
import shutil
import random

# ── PATHS ──────────────────────────────────────────────────────────────────
IMAGES_SRC   = r"C:\Users\apps\Downloads\labeling_images"
LABELS_SRC   = r"C:\Users\apps\Downloads\label_export\labels"
OUTPUT_BASE  = r"C:\Users\apps\Downloads\Week4\Dataset"

TRAIN_RATIO  = 0.8   # 80 train, 20 val  (out of 100 images)
RANDOM_SEED  = 42

# ── BUILD UUID → clean name MAP ────────────────────────────────────────────
label_map = {}   # clean stem → full original filename
for fname in os.listdir(LABELS_SRC):
    if fname.endswith(".txt"):
        # strip UUID prefix:  "03f47d65-frame_0042.txt" → "frame_0042"
        clean_stem = fname.split("-", 1)[-1].replace(".txt", "")
        label_map[clean_stem] = fname

# ── COLLECT ALL 100 IMAGES ─────────────────────────────────────────────────
all_images = sorted([f for f in os.listdir(IMAGES_SRC) if f.endswith(".png")])

labeled   = [f for f in all_images if os.path.splitext(f)[0] in label_map]
unlabeled = [f for f in all_images if os.path.splitext(f)[0] not in label_map]

print(f"Total images   : {len(all_images)}")
print(f"Labeled        : {len(labeled)}")
print(f"Unlabeled/test : {len(unlabeled)}")

# ── SPLIT LABELED → TRAIN / VAL ────────────────────────────────────────────
random.seed(RANDOM_SEED)
random.shuffle(labeled)
n_train = int(len(labeled) * TRAIN_RATIO)
train_imgs = labeled[:n_train]
val_imgs   = labeled[n_train:]

print(f"Train          : {len(train_imgs)}")
print(f"Val            : {len(val_imgs)}")
print(f"Test           : {len(unlabeled)}")

# ── CREATE FOLDER STRUCTURE ────────────────────────────────────────────────
folders = [
    "images/train", "images/val", "images/test",
    "labels/train", "labels/val"
]
for f in folders:
    os.makedirs(os.path.join(OUTPUT_BASE, f), exist_ok=True)

# ── COPY HELPER ─────────────────────────────────────────────────────────────
def copy_image(fname, split):
    src = os.path.join(IMAGES_SRC, fname)
    dst = os.path.join(OUTPUT_BASE, "images", split, fname)
    shutil.copy2(src, dst)

def copy_label(img_fname, split):
    stem = os.path.splitext(img_fname)[0]
    orig_label = label_map[stem]
    src = os.path.join(LABELS_SRC, orig_label)
    dst = os.path.join(OUTPUT_BASE, "labels", split, stem + ".txt")  # clean name
    shutil.copy2(src, dst)

# ── COPY FILES ──────────────────────────────────────────────────────────────
for f in train_imgs:
    copy_image(f, "train")
    copy_label(f, "train")

for f in val_imgs:
    copy_image(f, "val")
    copy_label(f, "val")

for f in unlabeled:
    copy_image(f, "test")

# ── WRITE train.txt & val.txt ───────────────────────────────────────────────
with open(os.path.join(OUTPUT_BASE, "train.txt"), "w") as fp:
    for f in train_imgs:
        fp.write(f"./images/train/{f}\n")

with open(os.path.join(OUTPUT_BASE, "val.txt"), "w") as fp:
    for f in val_imgs:
        fp.write(f"./images/val/{f}\n")

# ── WRITE dataset.yaml ──────────────────────────────────────────────────────
yaml_content = f"""path: {OUTPUT_BASE.replace(chr(92), "/")}
train: ./images/train
val:   ./images/val
test:  ./images/test

nc: 2
names:
  0: normal_drip
  1: anomaly_drip
"""
with open(os.path.join(OUTPUT_BASE, "dataset.yaml"), "w") as fp:
    fp.write(yaml_content)

print("\n✅ Dataset organized successfully!")
print(f"   Output → {OUTPUT_BASE}")