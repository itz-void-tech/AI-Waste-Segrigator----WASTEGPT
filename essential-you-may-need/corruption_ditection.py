import os
from PIL import Image
from tqdm import tqdm

dataset_path = r"G:\AI\Dataset_waste" #Enter your directory path here

print("Scanning for corrupted images...\n")

all_files = []

for root, dirs, files in os.walk(dataset_path):
    for file in files:
        all_files.append(os.path.join(root, file))

bad_images = 0
total_images = 0

for file_path in tqdm(all_files):

    # Skip non-image formats like SVG, GIF immediately
    #You can add more formats if needed
    if not file_path.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
        os.remove(file_path)
        print(f"Removed unsupported file: {file_path}")
        bad_images += 1
        continue

    try:
        with Image.open(file_path) as img:
            img.verify()
        total_images += 1

    except Exception:
        print(f"Deleting corrupted file: {file_path}")
        os.remove(file_path)
        bad_images += 1

print("\n=================================")
print(f"Total images checked: {total_images}")
print(f"Corrupted images removed: {bad_images}")
print("=================================")