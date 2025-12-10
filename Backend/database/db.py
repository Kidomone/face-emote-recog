import oracledb
import os


def get_connection():
    user = os.getenv("ORA_USER")
    password = os.getenv("ORA_PASS")
    dsn = os.getenv("ORA_DSN")

    if not user:
        raise ValueError("ORA_USER environment variable not set")
    if not password:
        raise ValueError("ORA_PASS environment variable not set")
    if not dsn:
        raise ValueError("ORA_DSN environment variable not set")

    return oracledb.connect(user=user, password=password, dsn=dsn)
