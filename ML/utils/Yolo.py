from ultralytics import YOLO
import cv2


class Yolo:
    """
    Creates Iterator for get faces from video with coords
    
    args: conf_treshhold=0.5, model_path="models/Yolo_face_detection.pt", camera_index=0
    
    DON'T FORGET TO CLOSE with Yolo.close()
    """
    def __init__(self, conf_treshhold=0.5, model_path="models/Yolo_face_detection.pt", camera_index=0):
        self.conf_treshhold = conf_treshhold
        self.cap = cv2.VideoCapture(camera_index)
        self.model = YOLO(model_path) 

    def __iter__(self):
        return self

    def __next__(self):
        """
        returns the following structure
        {  "num_faces": len(faces), 
            "faces": faces  }
        
        where faces is a list:
        faces.append({ "face_img": face_img,
                        "bbox": [x1, y1, x2, y2],
                        "confidence": conf  })
        """

        ret, frame = self.cap.read()
        if not ret:
            self.cap.release()
            raise StopIteration("Camera feed ended")

        results = self.model(frame)

        if results and results[0].boxes is not None and len(results[0].boxes) > 0:
            boxes = results[0].boxes
            faces = []
            
            for box in boxes:
                conf = float(box.conf[0])
                if conf < self.conf_treshhold:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                face_img = frame[y1:y2, x1:x2]

                faces.append({
                    "face_img": face_img,
                    "bbox": [x1, y1, x2, y2],
                    "confidence": conf
                })
            
            return {
                "num_faces": len(faces),
                "faces": faces
            }
        return {
                "num_faces": 0,
                "faces": []
            }
    

    def close(self):
        self.cap.release()
