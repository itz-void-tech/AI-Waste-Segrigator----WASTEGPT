import os
import random
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import accuracy_score
from tqdm import tqdm

# =====================================
# 🔧 DATASET PATH
# =====================================
dataset_path = r"G:\AI\Dataset_waste"

# =====================================
# ⚙️ TRAINING SETTINGS
# =====================================
batch_size = 32
epochs = 20
learning_rate = 0.0003
image_size = 224
model_save_path = "waste_classifier.pth"
class_file_path = "class_names.json"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n🔥 Using device: {device}")

# =====================================
# 🔥 TRANSFORMS
# =====================================
train_transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomResizedCrop(image_size),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# =====================================
# 📂 LOAD DATASET (WITHOUT TRANSFORM FIRST)
# =====================================
base_dataset = datasets.ImageFolder(dataset_path)

class_names = base_dataset.classes
num_classes = len(class_names)

print("\n📌 Classes Found:")
for cls in class_names:
    print(" -", cls)

print("\n📊 Image Count Per Class:")
for cls in class_names:
    folder = os.path.join(dataset_path, cls)
    print(f"{cls}: {len(os.listdir(folder))} images")

# Save class names
with open(class_file_path, "w") as f:
    json.dump(class_names, f)

# =====================================
# 📊 TRAIN / VAL SPLIT (80/20)
# =====================================
train_size = int(0.8 * len(base_dataset))
val_size = len(base_dataset) - train_size

train_dataset, val_dataset = random_split(
    base_dataset, [train_size, val_size]
)

# Apply transforms separately
train_dataset.dataset.transform = train_transforms
val_dataset.dataset.transform = val_transforms

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

print(f"\n📦 Total Images: {len(base_dataset)}")
print(f"🚀 Training Images: {len(train_dataset)}")
print(f"🔍 Validation Images: {len(val_dataset)}")

# Show sample training files
print("\n📁 Sample Training Files:")
for i in range(5):
    print(train_dataset.dataset.samples[train_dataset.indices[i]][0])

# =====================================
# 🧠 MODEL (EfficientNet-B0)
# =====================================
model = models.efficientnet_b0(weights="DEFAULT")
model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
model = model.to(device)

# =====================================
# 🧪 LOSS + OPTIMIZER
# =====================================
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

# =====================================
# 🚀 TRAINING LOOP
# =====================================
best_accuracy = 0

epoch_bar = tqdm(range(epochs), desc="Training Progress")

for epoch in epoch_bar:

    model.train()
    running_loss = 0

    train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)

    for images, labels in train_bar:

        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        train_bar.set_postfix(loss=loss.item())

    scheduler.step()

    # ======================
    # VALIDATION
    # ======================
    model.eval()
    all_preds = []
    all_labels = []

    val_bar = tqdm(val_loader, desc="Validating", leave=False)

    with torch.no_grad():
        for images, labels in val_bar:

            images = images.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1).cpu()

            all_preds.extend(preds.numpy())
            all_labels.extend(labels.numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    avg_loss = running_loss / len(train_loader)
    current_lr = optimizer.param_groups[0]['lr']

    print(f"\n📌 Epoch [{epoch+1}/{epochs}]")
    print(f"📉 Avg Loss: {avg_loss:.4f}")
    print(f"🎯 Validation Accuracy: {accuracy:.4f}")
    print(f"📚 Learning Rate: {current_lr}")

    # Save best model
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        torch.save(model.state_dict(), model_save_path)
        print("🔥 Best model saved!")

print("\n=================================")
print("✅ Training Complete")
print(f"🏆 Best Accuracy: {best_accuracy:.4f}")
print("📁 Model Saved As:", model_save_path)
print("📁 Class Names Saved As:", class_file_path)
print("=================================")