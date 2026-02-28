import torch
import torch.nn as nn
from torchvision import models, transforms
import cv2
from PIL import Image
import numpy as np

# =============================
# CONFIG
# =============================
MODEL_PATH = "waste_classifier_best.pth"

class_names = [
    "e-waste-bin",
    "general-waste-bin",
    "hazardous-bin",
    "organic-bin",
    "recyclable-bin"
]

object_map = {
    "e-waste-bin": "Electronic Waste",
    "general-waste-bin": "General Waste Item",
    "hazardous-bin": "Hazardous Material",
    "organic-bin": "Organic Waste",
    "recyclable-bin": "Recyclable Item"
}

confidence_threshold = 85

# =============================
# DEVICE
# =============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =============================
# LOAD MODEL
# =============================
model = models.efficientnet_b0(weights=None)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 5)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# =============================
# WEBCAM
# =============================
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    box_size = 300
    cx, cy = w // 2, h // 2
    x1 = cx - box_size // 2
    y1 = cy - box_size // 2
    x2 = cx + box_size // 2
    y2 = cy + box_size // 2

    crop = frame[y1:y2, x1:x2]

    predicted_object = "None"
    predicted_bin = "None"
    confidence = 0

    try:
        pil_img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        input_tensor = transform(pil_img).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(input_tensor)
            probs = torch.softmax(output, dim=1)
            conf, pred = torch.max(probs, 1)

        confidence = conf.item() * 100

        if confidence >= confidence_threshold:
            predicted_bin = class_names[pred.item()]
            predicted_object = object_map[predicted_bin]

    except:
        pass

    # =============================
    # STYLISH UI PANEL
    # =============================
    overlay = frame.copy()
    cv2.rectangle(overlay, (20, 20), (420, 140), (0, 0, 0), -1)
    alpha = 0.6
    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    # Border for center detection area
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)

    # Text Styling
    color = (0,255,0) if predicted_object != "None" else (0,0,255)

    cv2.putText(frame,
                f"Object: {predicted_object}",
                (40, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2)

    cv2.putText(frame,
                f"Bin: {predicted_bin}",
                (40, 95),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2)

    cv2.putText(frame,
                f"Confidence: {confidence:.1f}%",
                (40, 130),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255,255,255),
                2)

    cv2.imshow("Smart Waste Bin", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()