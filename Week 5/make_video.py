import subprocess
import os

folder = r"C:\Users\apps\Downloads\Week4\runs\detect\predict-2"
output = r"C:\Users\apps\Downloads\Week4\W5_detected_output.mp4"

# Write a list file for ffmpeg
list_file = r"C:\Users\apps\Downloads\Week4\framelist.txt"
files = sorted([f for f in os.listdir(folder) if f.endswith(".jpg")])

with open(list_file, "w") as f:
    for fname in files:
        f.write(f"file '{os.path.join(folder, fname)}'\n")
        f.write("duration 0.1\n")  # 10 fps = 0.1s per frame

subprocess.run([
    "ffmpeg", "-y", "-f", "concat", "-safe", "0",
    "-i", list_file,
    "-c:v", "libx264", "-pix_fmt", "yuv420p",
    output
])

print("✅ Video created:", output)