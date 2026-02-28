import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import sys

# ==============================
# 🔧 CHANGE THIS PATH
# ==============================
MODEL_PATH = "waste_classifier_best.pth"
IMAGE_PATH = "pen.jfif"    #change to your test image

# ==============================
# 🗂 CLASS NAMES (5 BINS)
# ==============================
class_names = [
    "e-waste-bin",
    "general-waste-bin",
    "hazardous-bin",
    "organic-bin",
    "recyclable-bin"
]

# ==============================
# DEVICE
# ==============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==============================
# LOAD MODEL
# ==============================
model = models.efficientnet_b0(weights=None)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 5)

model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

print("✅ Model loaded successfully")

# ==============================
# TRANSFORM (must match training)
# ==============================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Lambda(lambda img: img.convert("RGB")),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ==============================
# LOAD IMAGE
# ==============================
image = Image.open(IMAGE_PATH)
input_tensor = transform(image).unsqueeze(0).to(device)

# ==============================
# PREDICTION
# ==============================
with torch.no_grad():
    outputs = model(input_tensor)
    probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
    confidence, predicted = torch.max(probabilities, 0)

predicted_class = class_names[predicted.item()]
confidence_percent = confidence.item() * 100

# ==============================
# RESULT
# ==============================
print(f"\n🔍 Predicted Bin: {predicted_class}")
print(f"🔥 Confidence: {confidence_percent:.2f}%")

# ==============================
# SHOW IMAGE
# ==============================
plt.imshow(image)
plt.title(f"{predicted_class} ({confidence_percent:.2f}%)")
plt.axis("off")
plt.show()