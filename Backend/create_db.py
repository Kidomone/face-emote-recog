import oracledb
import numpy as np

connection = oracledb.connect(user="system", password='admin', dsn="localhost/xepdb1")
cursor = connection.cursor()
cursor.execute("""
BEGIN
    EXECUTE IMMEDIATE 'CREATE TABLE PERSONS (
        PERSON_ID NUMBER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
        FULL_NAME VARCHAR2(200) NOT NULL,
        EMAIL VARCHAR2(100),
        CREATED_AT DATE DEFAULT SYSDATE
    )';
EXCEPTION
    WHEN OTHERS THEN
        IF SQLCODE != -955 THEN
            RAISE;
        END IF;
END;""")

cursor.execute("""
BEGIN
    EXECUTE IMMEDIATE 'CREATE TABLE FACE_VECTORS (
        VECTOR_ID NUMBER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
        PERSON_ID NUMBER REFERENCES PERSONS(PERSON_ID) ON DELETE CASCADE,
        VECTOR_DATA BLOB,
        VECTOR_SIZE NUMBER,
        CREATED_AT DATE DEFAULT SYSDATE
    )';
EXCEPTION
    WHEN OTHERS THEN
        IF SQLCODE != -955 THEN
            RAISE;
        END IF;
END;""")

connection.commit()