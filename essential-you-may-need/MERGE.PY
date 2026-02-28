import os
import shutil

# 🔧 CHANGE THESE PATHS ONLY
source_parent = r"G:\AI\needmerging"
destination   = r"G:\AI\WasteDataset"


def get_unique_path(dest_file):
    if not os.path.exists(dest_file):
        return dest_file

    base, ext = os.path.splitext(dest_file)
    counter = 1

    while True:
        new_file = f"{base}_{counter}{ext}"
        if not os.path.exists(new_file):
            return new_file
        counter += 1


print("Starting folder-wise merge...")
os.makedirs(destination, exist_ok=True)

total = 0

# 🔥 Loop through each CLASS folder
for class_folder in os.listdir(source_parent):

    src_class_path = os.path.join(source_parent, class_folder)

    if not os.path.isdir(src_class_path):
        continue

    print(f"Processing: {class_folder}")

    # Create SAME folder in destination
    dest_class_path = os.path.join(destination, class_folder)
    os.makedirs(dest_class_path, exist_ok=True)

    for root, dirs, files in os.walk(src_class_path):
        for file in files:

            src_file = os.path.join(root, file)
            dest_file = os.path.join(dest_class_path, file)

            dest_file = get_unique_path(dest_file)

            shutil.copy2(src_file, dest_file)
            total += 1

print("\n==============================")
print("✅ DONE — Folders preserved")
print(f"📦 Total files copied: {total}")
print("==============================")