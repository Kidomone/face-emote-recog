import oracledb
import os

dsn = os.getenv("ORA_DSN", "localhost/xepdb1")

connection = oracledb.connect(
    user=os.getenv("ORA_USER", "system"),
    password=os.getenv("ORA_PASS", "admin"),
    dsn=dsn,
)
cursor = connection.cursor()

cursor.execute(
    """
BEGIN
    EXECUTE IMMEDIATE 'CREATE TABLE USER_PHOTOS (
        ID NUMBER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
        UUID VARCHAR2(36) UNIQUE,
        USER_ID VARCHAR2(36) NOT NULL,
        HUMAN_ID VARCHAR2(36) NOT NULL,
        PHOTO_URL VARCHAR2(1000) NOT NULL,
        YOLO_FACE_BBOX CLOB,
        IS_PRIMARY NUMBER(1) DEFAULT 0 CHECK (IS_PRIMARY IN (0, 1)),
        CREATED_AT TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        STATUS_CODE VARCHAR2(50)
    )';
EXCEPTION
    WHEN OTHERS THEN
        IF SQLCODE != -955 THEN
            RAISE;
        END IF;
END;"""
)

connection.commit()
cursor.close()
connection.close()
