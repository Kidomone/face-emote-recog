from dotenv import load_dotenv

load_dotenv()

from db import get_connection

tables_to_drop = [
    "FACE_DETECTIONS",
    "UPLOADED_IMAGES",
    "USER_PHOTOS",
    "HUMANS",
    "USERS",
]


def drop_tables():
    conn = get_connection()
    cur = conn.cursor()
    for table in tables_to_drop:
        try:
            cur.execute(f"DROP TABLE {table} CASCADE CONSTRAINTS")
            print(f"Таблица {table} удалена")
        except Exception as e:
            print(f"Не удалось удалить таблицу {table}: {e}")
    conn.commit()
    cur.close()
    conn.close()
    print("Удаление таблиц завершено.")


if __name__ == "__main__":
    drop_tables()
