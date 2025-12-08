import oracledb

connection = oracledb.connect(user="system", password='admin', dsn="localhost/xepdb1")
cursor = connection.cursor()

cursor.execute("""
BEGIN
    EXECUTE IMMEDIATE 'CREATE TABLE USER_PHOTOS (
        ID NUMBER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
        UUID VARCHAR2(36),
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
END;""")

connection.commit()
print("Таблица USER_PHOTOS создана или уже существует")

cursor.close()
connection.close()
