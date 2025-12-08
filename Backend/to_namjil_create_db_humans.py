import oracledb

connection = oracledb.connect(user="system", password='admin', dsn="localhost/xepdb1")
cursor = connection.cursor()

cursor.execute("""
BEGIN
    EXECUTE IMMEDIATE 'CREATE TABLE HUMANS (
        ID NUMBER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
        UUID VARCHAR2(36),
        USER_ID VARCHAR2(36),
        FIRST_NAME VARCHAR2(255),
        LAST_NAME VARCHAR2(255),
        KNOWN_FACE_URL VARCHAR2(500),
        CREATED_AT TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        UPDATED_AT TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )';
EXCEPTION
    WHEN OTHERS THEN
        IF SQLCODE != -955 THEN
            RAISE;
        END IF;
END;""")

connection.commit()

cursor.close()
connection.close()
