import cv2
import numpy as np
from deepface import DeepFace
from sklearn.metrics.pairwise import cosine_similarity
from api.database.db import get_connection
import uuid


class FaceRecognizer:
    def __init__(
        self, model_name="Facenet", detector_backend="opencv", test_mode=False
    ):
        self.model_name = model_name
        self.detector_backend = detector_backend
        self.test_mode = test_mode
        self.embeddings_db = {}

        if test_mode:
            print(f"[TEST MODE] FaceRecognizer initialized (model={model_name})")

        self.load_embeddings_from_db()

    def _generate_stub_embedding(self, dimensions=512):
        return np.random.randn(dimensions).astype(np.float32)

    def load_embeddings_from_db(self):
        """Load all embeddings from FACE_DETECTIONS / HUMANS table into memory"""
        conn = get_connection()
        cur = conn.cursor()
        self.embeddings_db = {}

        try:
            cur.execute(
                """
                SELECT h.UUID, fd.DETECTED_BBOX, fd.UUID
                FROM HUMANS h
                LEFT JOIN FACE_DETECTIONS fd ON fd.DETECTED_HUMAN_ID = h.UUID
            """
            )
            rows = cur.fetchall()
            for row in rows:
                human_uuid = row[0]
                # For now, we need to store embedding separately in the DB.
                # Assume we have a CLOB column or BLOB to store embeddings
                # Here we skip embedding reconstruction from FACE_DETECTIONS
                # We'll load as they are added dynamically
                pass
        except Exception as e:
            print(f"Error loading embeddings from DB: {e}")
        finally:
            cur.close()
            conn.close()

    def extract_embedding_from_roi(self, face_roi):
        if self.test_mode:
            return self._generate_stub_embedding(512)

        try:
            rgb_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)

            result = DeepFace.represent(
                rgb_face,
                model_name=self.model_name,
                detector_backend="skip",
                enforce_detection=False,
            )

            if result:
                return np.array(result[0]["embedding"])
            return None

        except Exception as e:
            print(f"Error extracting embedding from ROI: {e}")
            return None

    def recognize_face(self, embedding, threshold=0.45):
        if not self.embeddings_db:
            return "unknown", 0.0

        best_uuid = "unknown"
        best_score = 0.0
        embedding = np.array(embedding)

        for human_uuid, stored_emb in self.embeddings_db.items():
            sim = cosine_similarity([embedding], [stored_emb])[0][0]
            if sim > best_score:
                best_score = sim
                best_uuid = human_uuid

        if best_score < threshold:
            return "unknown", best_score

        return best_uuid, best_score

    def add_new_human(self, embedding):
        human_uuid = str(uuid.uuid4())

        conn = get_connection()
        cur = conn.cursor()
        try:
            cur.execute("INSERT INTO HUMANS (UUID) VALUES (:1)", (human_uuid,))
            conn.commit()
            self.embeddings_db[human_uuid] = embedding
            info_message = f"Добавлен незнакомый человек"
            return human_uuid, info_message
        except Exception as e:
            print(f"Error inserting new human: {e}")
            return None, f"Error adding new human: {e}"
        finally:
            cur.close()
            conn.close()

    def process_roi(self, face_roi):
        embedding = self.extract_embedding_from_roi(face_roi)
        if embedding is None:
            return "unknown", 0.0, None, "Failed to extract embedding"

        human_uuid, score = self.recognize_face(embedding)
        info_message = "Это знакомый человек"

        if human_uuid == "unknown":
            human_uuid, info_message = self.add_new_human(embedding)
            score = 100.0

        return human_uuid, score, embedding, info_message
