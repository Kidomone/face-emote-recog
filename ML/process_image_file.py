import cv2
import torch
import numpy as np
from PIL import Image
from ultralytics import YOLO
import torchvision.transforms.v2 as T
from io import BytesIO


from ML.utils.Custom_models import Resnet_Custom


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


# Load models (do this once at module level)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_detect = YOLO("ML/models/Yolo_face_detection.pt")

model_emotion = Resnet_Custom(output_shape=len(affectnet_labels_names))
pack = torch.load("ML/models/Resnet_Custom_best_f1.pth", map_location=device, weights_only=False)
model_emotion.load_state_dict(pack)
model_emotion.eval()
model_emotion = model_emotion.to(device)

# Transforms for emotion recognition
transforms = T.Compose([
    T.ToImage(),
    T.ToDtype(torch.float32, scale=True),
    T.Normalize(mean=[0.5], std=[0.5]),
])

# Emotion labels


def process_image(image_bytes: bytes) -> tuple[bytes, dict]:
    """
    Process image: detect faces and their emotions
    
    Args:
        image_bytes: Input image as bytes
        
    Returns:
        tuple: (processed_image_bytes, api_data)
            - processed_image_bytes: Image with annotations
            - api_data: Dict with detection results
    """
    # Convert bytes to cv2 image
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        raise ValueError("Failed to decode image")
    
    faces_data = []
    
    try:
        # Detect faces
        results = results = model_detect(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), conf=0.2)

        
        if results and results[0].boxes is not None:
            boxes = results[0].boxes
            
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # Crop face
                face = img[y1:y2, x1:x2]
                
                if face.size == 0:
                    continue
                
                # Convert to grayscale and prepare for emotion model
                face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                face_pil = Image.fromarray(face_gray)
                face_tensor = transforms(face_pil).unsqueeze(0).to(device)
                
                # Predict emotion
                with torch.no_grad():
                    outputs = model_emotion(face_tensor)
                    probs = torch.softmax(outputs, dim=1)
                    pred_idx = torch.argmax(probs, dim=1).item()
                    confidence = probs[0][pred_idx].item()
                
                emotion = affectnet_labels_names[pred_idx]
                
                # Draw on image
                cv2.rectangle(img, (x1, y1), (x2, y2), (247, 0, 255), 2)
                text = f"{emotion}: {confidence:.2f}"
                cv2.putText(img, text, (x1, y1 - 4),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (247, 0, 255), 2)
                
                # Store data
                faces_data.append({
                    "bbox": [x1, y1, x2, y2],
                    "emotion": emotion,
                    "confidence": float(confidence),
                    "all_scores": {
                        label: float(probs[0][i].item())
                        for i, label in enumerate(affectnet_labels_names)
                    }
                })
    
    except Exception as e:
        print(f"Error processing image: {e}")
        # Return original image if processing fails
        pass
    
    # Encode processed image back to bytes
    _, buffer = cv2.imencode('.jpg', img)
    processed_bytes = buffer.tobytes()
    
    # Prepare API data
    api_data = {
        "faces_detected": len(faces_data),
        "faces": faces_data
    }
    
    return processed_bytes, api_data