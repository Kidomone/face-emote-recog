import oracledb
import numpy as np

connection = oracledb.connect(user="system", password='admin', dsn="localhost/xepdb1")
cursor = connection.cursor()



# Вставка нового человека
cursor.execute("INSERT INTO PERSONS (FULL_NAME, EMAIL) VALUES (:1, :2)", ("Иван Иванов", "ivan@example.com"))
connection.commit()

# Получаем ID
cursor.execute("SELECT PERSON_ID FROM PERSONS WHERE EMAIL = :1", ("ivan@example.com",))
person_id = cursor.fetchone()[0]

# Создаём вектор лица
vector = np.random.rand(128).astype(np.float32)
print(vector)
vector_bytes = vector.tobytes()
print(vector_bytes)
cursor.execute("INSERT INTO FACE_VECTORS (PERSON_ID, VECTOR_DATA, VECTOR_SIZE) VALUES (:1, :2, :3)",
               (person_id, vector_bytes, len(vector)))
connection.commit()

# Проверяем вставку
cursor.execute("SELECT VECTOR_DATA FROM FACE_VECTORS WHERE PERSON_ID = :1", (person_id,))
blob, = cursor.fetchone()
vector_from_db = np.frombuffer(blob.read(), dtype=np.float32)
print(vector_from_db)
