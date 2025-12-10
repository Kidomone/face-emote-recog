import oracledb
import os
import numpy as np

dsn = os.getenv("ORA_DSN", "localhost/xepdb1")

connection = oracledb.connect(
    user=os.getenv("ORA_USER", "system"),
    password=os.getenv("ORA_PASS", "admin"),
    dsn=dsn,
)
cursor = connection.cursor()

cursor.execute(
    "INSERT INTO PERSONS (FULL_NAME, EMAIL) VALUES (:1, :2)",
    ("Иван Иванов", "ivan@example.com"),
)
connection.commit()

cursor.execute("SELECT PERSON_ID FROM PERSONS WHERE EMAIL = :1", ("ivan@example.com",))
person_id = cursor.fetchone()[0]

vector = np.random.rand(128).astype(np.float32)
vector_bytes = vector.tobytes()
cursor.execute(
    "INSERT INTO FACE_VECTORS (PERSON_ID, VECTOR_DATA, VECTOR_SIZE) VALUES (:1, :2, :3)",
    (person_id, vector_bytes, len(vector)),
)
connection.commit()

cursor.execute(
    "SELECT VECTOR_DATA FROM FACE_VECTORS WHERE PERSON_ID = :1", (person_id,)
)
(blob,) = cursor.fetchone()
vector_from_db = np.frombuffer(blob.read(), dtype=np.float32)
print(vector_from_db)
