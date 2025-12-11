import cv2
import torch
import numpy as np
from PIL import Image
from ultralytics import YOLO
import torchvision.transforms.v2 as T
import os
import uuid

from ML.utils.Custom_models import Resnet_Custom
from ML.utils.FaceRecognizer import FaceRecognizer

affectnet_labels_names = [
    "Anger",
    "Contempt",
    "Disgust",
    "Fear",
    "Happy",
    "Neutral",
    "Sad",
    "Surprise",
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_detect = YOLO("ML/models/Yolo_face_detection.pt")

model_emotion = Resnet_Custom(output_shape=len(affectnet_labels_names))
model_emotion.eval()
model_emotion = model_emotion.to(device)

face_recognizer = FaceRecognizer(
    model_name="Facenet", detector_backend="opencv", test_mode=False
)

transforms = T.Compose(
    [
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=[0.5], std=[0.5]),
    ]
)


def process_image(image_bytes: bytes) -> tuple[bytes, dict, str]:
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Failed to decode image")

    faces_data = []

    try:
        results = model_detect(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), conf=0.2)

        if results and results[0].boxes is not None:
            boxes = results[0].boxes

            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                face = img[y1:y2, x1:x2]
                if face.size == 0:
                    continue

                # Emotion prediction
                face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                face_pil = Image.fromarray(face_gray)
                face_tensor = transforms(face_pil).unsqueeze(0).to(device)

                with torch.no_grad():
                    outputs = model_emotion(face_tensor)
                    probs = torch.softmax(outputs, dim=1)
                    pred_idx = torch.argmax(probs, dim=1).item()
                    emotion_conf = float(probs[0][pred_idx].item())
                    emotion = affectnet_labels_names[pred_idx]

                # Face recognition + info message
                human_uuid, identity_conf, embedding, info_message = (
                    face_recognizer.process_roi(face)
                )

                # Draw annotations
                cv2.rectangle(img, (x1, y1), (x2, y2), (247, 0, 255), 2)
                label_text = f"{human_uuid} | {emotion} ({emotion_conf:.2f})"
                cv2.putText(
                    img,
                    label_text,
                    (x1, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (247, 0, 255),
                    2,
                )

                faces_data.append(
                    {
                        "bbox": [x1, y1, x2, y2],
                        "emotion": emotion,
                        "emotion_confidence": emotion_conf,
                        "identity": human_uuid,
                        "identity_confidence": float(identity_conf),
                        "info_message": info_message,
                        "all_emotion_scores": {
                            label: float(probs[0][i].item())
                            for i, label in enumerate(affectnet_labels_names)
                        },
                    }
                )

    except Exception as e:
        print(f"Error processing image: {e}")

    _, buffer = cv2.imencode(".jpg", img)
    processed_bytes = buffer.tobytes()
    api_data = {"faces_detected": len(faces_data), "faces": faces_data}

    return processed_bytes, api_data
