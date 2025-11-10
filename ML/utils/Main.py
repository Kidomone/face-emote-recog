from FaceRecognizer import FaceRecognizer

recognizer = FaceRecognizer()

test_names = ["person1", "person2", "person3"]
recognizer.generate_test_database(test_names, "test_db.txt")
recognizer.load_embeddings_from_file("test_db.txt")
recognizer.analyze_embeddings()

test_embedding = recognizer._generate_stub_embedding()
name_test, similarity_test = recognizer.recognize_face(test_embedding)
print(f"Тест на рандомных данных: {name_test} (сходство: {similarity_test:.3f})")

if recognizer.embeddings_db:
    existing_name = list(recognizer.embeddings_db.keys())[0]
    existing_embedding = recognizer.embeddings_db[existing_name]
    name_exist, similarity_exist = recognizer.recognize_face(existing_embedding)
    print(
        f"Тест с существующим эмбеддингом: {name_exist} (сходство: {similarity_exist:.3f})"
    )


# Как-то так связать модельки, например
"""recognizer = FaceRecognizer()
recognizer.load_embeddings_from_file("face_database.txt")

yolo_detector = YOLODetector()  <--- Из класса YOLO будем получать bbox'ы лиц
frame = cv2.imread("test_image.jpg")

yolo_results = yolo_detector.detect_faces(frame)  <--- А затем детектить :)
combined_results = recognizer.process_yolo_detections(frame, yolo_results)  <--- И после детекции отправлять в модельку DeepFace

for result in combined_results:
    x, y, w, h = result['bbox']
    label = f"{result['identity']} ({result['emotion']}) {result['confidence']:.2f}"
    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
"""
