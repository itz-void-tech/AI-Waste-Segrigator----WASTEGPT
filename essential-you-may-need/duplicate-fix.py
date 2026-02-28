import os
import hashlib
import uuid

# ==========================
# 🔧 CHANGE THIS PATH ONLY
# ==========================
dataset_path = r"G:\AI\WasteDataset"


def get_file_hash(file_path):
    hasher = hashlib.sha256()
    with open(file_path, 'rb') as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
    return hasher.hexdigest()


def clean_and_rename_dataset(base_path):

    print("Starting dataset cleanup...\n")

    total_deleted = 0
    total_renamed = 0

    for folder in os.listdir(base_path):

        folder_path = os.path.join(base_path, folder)

        if not os.path.isdir(folder_path):
            continue

        print(f"Processing folder: {folder}")

        seen_hashes = {}
        temp_files = []

        # Step 1 — Remove duplicates
        for file in os.listdir(folder_path):

            file_path = os.path.join(folder_path, file)

            if not os.path.isfile(file_path):
                continue

            try:
                file_hash = get_file_hash(file_path)
            except:
                continue

            if file_hash in seen_hashes:
                os.remove(file_path)
                total_deleted += 1
                print(f"Deleted duplicate: {file}")
            else:
                seen_hashes[file_hash] = file_path

        # Step 2 — Temporary rename (avoid collisions)
        for file in os.listdir(folder_path):

            old_path = os.path.join(folder_path, file)

            if not os.path.isfile(old_path):
                continue

            ext = os.path.splitext(file)[1].lower()
            temp_name = f"temp_{uuid.uuid4().hex}{ext}"
            temp_path = os.path.join(folder_path, temp_name)

            os.rename(old_path, temp_path)
            temp_files.append(temp_path)

        # Step 3 — Final clean rename
        temp_files = sorted(temp_files)

        for index, temp_path in enumerate(temp_files, start=1):

            ext = os.path.splitext(temp_path)[1].lower()
            final_name = f"{folder}_{index:04d}{ext}"
            final_path = os.path.join(folder_path, final_name)

            os.rename(temp_path, final_path)
            total_renamed += 1

        print(f"Finished folder: {folder}\n")

    print("=================================")
    print("✅ DATASET CLEANED SUCCESSFULLY")
    print(f"🗑️ Total duplicates deleted: {total_deleted}")
    print(f"✏️ Total files renamed: {total_renamed}")
    print("=================================")


if __name__ == "__main__":
    clean_and_rename_dataset(dataset_path)