from deepface import DeepFace
import cv2
import numpy as np
import os
from sklearn.metrics.pairwise import cosine_similarity


class FaceRecognizer:
    def __init__(
        self, model_name="Facenet", detector_backend="opencv", test_mode=False
    ):
        """
        Инициализация с предобученной моделью из библиотеки DeepFace
        Доступные модели: 'VGG-Face', 'Facenet', 'OpenFace', 'DeepFace', 'DeepID', 'ArcFace' и другие
        Детекторы: 'opencv', 'mtcnn', 'retinaface'
        """
        self.model_name = model_name
        self.detector_backend = detector_backend
        self.embeddings_db = {}
        self.test_mode = test_mode

        if test_mode:
            print(f"Активирован тестовый режим (модель: {model_name})")

    def _generate_stub_embedding(self, dimensions=512):
        return np.random.randn(dimensions).astype(np.float32)

    def generate_test_database(self, names_list, output_file="test_database.txt"):
        with open(output_file, "w", encoding="utf-8") as f:
            for name in names_list:
                embedding = self._generate_stub_embedding()
                vector_str = " ".join([f"{x:.8f}" for x in embedding])
                f.write(f"{name}: [{vector_str}]\n")
        print(f"Сгенерирована тестовая база эмбеддингов с {len(names_list)} записями")

    def extract_embedding(self, image_path):
        if self.test_mode:
            print(f"[Тест] Генерация случайного эмбеддинга")
            embedding = self._generate_stub_embedding(128)
            face_area = {"x": 100, "y": 100, "w": 200, "h": 200}
            return embedding, face_area

        try:
            result = DeepFace.represent(
                img_path=image_path,
                model_name=self.model_name,
                detector_backend=self.detector_backend,
                enforce_detection=False,
            )

            if result:
                embedding = result[0]["embedding"]
                face_area = result[0]["facial_area"]

                print(f"Эмбеддинг успешно извлечен, размерность: {len(embedding)}")
                return np.array(embedding), face_area
            else:
                print("Не удалось извлечь эмбеддинг")
                return None, None

        except Exception as e:
            print(f"Ошибка при извлечении эмбеддинга: {e}")
            return None, None

    def save_embedding_to_file(self, embedding, name, filename="face_embeddings.txt"):
        with open(filename, "a", encoding="utf-8") as f:
            vector_str = " ".join([f"{x:.8f}" for x in embedding])
            f.write(f"{name}: [{vector_str}]\n")
        print(f"Сохранен эмбеддинг для: {name}")

    def load_embeddings_from_file(self, filename="face_embeddings.txt"):
        self.embeddings_db = {}
        if os.path.exists(filename):
            with open(filename, "r", encoding="utf-8") as f:
                for line in f:
                    if ":" in line:
                        name, vector_str = line.strip().split(":", 1)
                        vector_str = vector_str.strip()[1:-1]
                        vector = np.fromstring(vector_str, sep=" ")
                        self.embeddings_db[name.strip()] = vector
            print(f"Загружено {len(self.embeddings_db)} эмбеддингов")
        return self.embeddings_db

    def recognize_face(self, embedding, threshold=0.5):
        if not self.embeddings_db:
            print("Нет базы данных эмбеддингов")
            return "unknown", 0.0

        best_match = "unknown"
        best_similarity = 0.0

        for name, stored_embedding in self.embeddings_db.items():
            similarity = cosine_similarity([embedding], [stored_embedding])[0][0]
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = name

        if best_similarity < threshold:
            return "unknown", best_similarity

        return best_match, best_similarity

    def verify_faces(self, img1_path, img2_path):
        try:
            result = DeepFace.verify(
                img1_path=img1_path,
                img2_path=img2_path,
                model_name=self.model_name,
                detector_backend=self.detector_backend,
            )

            print(f"Сравнение изображений {img1_path} и {img2_path}:")
            print(f"- verified: {result['verified']}")
            print(f"- distance: {result['distance']:.4f}")
            print(f"- confidence: {result['confidence']:.4f}")

            return result

        except Exception as e:
            print(f"Ошибка сравнения изображений: {e}")
            return None

    def build_database_from_folder(self, folder_path, output_file="face_database.txt"):
        if not os.path.exists(folder_path):
            print(f"Папки {folder_path} не существует")
            return

        open(output_file, "w").close()

        for person_name in os.listdir(folder_path):
            person_path = os.path.join(folder_path, person_name)

            if os.path.isdir(person_path):
                print(f"\nОбрабатываю человека: {person_name}")

                for img_file in os.listdir(person_path):
                    if img_file.lower().endswith((".jpg", ".jpeg", ".png")):
                        img_path = os.path.join(person_path, img_file)

                        try:
                            embedding = self.extract_embedding(img_path)
                            if embedding is not None:
                                self.save_embedding_to_file(
                                    embedding, person_name, output_file
                                )
                                break
                            else:
                                print(
                                    f"Не удалось извлечь эмбеддинг из файла {img_file}"
                                )
                        except Exception as e:
                            print(f"Ошибка с файлом {img_file}: {e}")

        self.load_embeddings_from_file(output_file)

    def analyze_embeddings(self):
        if not self.embeddings_db:
            print("База эмбеддингов пуста")
            return

        print("Анализ базы эмбеддингов:")
        print(f"Количество записей: {len(self.embeddings_db)}")

        first_key = list(self.embeddings_db.keys())[0]
        embedding_shape = self.embeddings_db[first_key].shape
        print(f"Размерность эмбеддингов: {embedding_shape}")
        print(f"Имена в базе: {list(self.embeddings_db.keys())}")

    def extract_embedding_from_roi(self, face_roi):
        try:
            if self.test_mode:
                return self._generate_stub_embedding(512)

            temp_path = "temp_roi.jpg"
            cv2.imwrite(temp_path, face_roi)

            result = DeepFace.represent(
                img_path=temp_path,
                model_name=self.model_name,
                detector_backend="skip",
                enforce_detection=False,
            )

            if result:
                return np.array(result[0]["embedding"])
            return None

        except Exception as e:
            print(f"Ошибка извлечения эмбеддинга из ROI: {e}")
            return None

    def process_yolo_detections(self, frame, yolo_detections):
        results = []

        if yolo_detections is None or len(yolo_detections) == 0:
            return results

        for detection in yolo_detections:
            x, y, w, h = detection["bbox"]
            emotion = detection.get("emotion", "unknown")

            face_roi = frame[y : y + h, x : x + w]

            if face_roi.size == 0:
                continue

            embedding = self.extract_embedding_from_roi(face_roi)

            if embedding is not None:
                identity, confidence = self.recognize_face(embedding)

                results.append(
                    {
                        "bbox": (x, y, w, h),
                        "emotion": emotion,
                        "identity": identity,
                        "confidence": float(confidence),
                        "embedding": embedding,
                    }
                )

        return results
