import oracledb

connection = oracledb.connect(user="system", password='admin', dsn="localhost/xepdb1")
cursor = connection.cursor()

cursor.execute("""
BEGIN
    EXECUTE IMMEDIATE 'CREATE TABLE USERS (
        ID NUMBER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
        UUID VARCHAR2(36),
        EMAIL VARCHAR2(255) NOT NULL,
        USERNAME VARCHAR2(255) NOT NULL,
        PASSWORD_HASH VARCHAR2(255) NOT NULL,
        FIRST_NAME VARCHAR2(255) NOT NULL,
        LAST_NAME VARCHAR2(255) NOT NULL,
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
