from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import asyncio
import base64
import json
import uuid
import os

from dotenv import load_dotenv

load_dotenv()

from backend.database.db import get_connection
from ML.process_image_file import process_image

app = FastAPI()

templates = Jinja2Templates(directory=os.path.join(os.getcwd(), "frontend"))
app.mount(
    "/static",
    StaticFiles(directory=os.path.join(os.getcwd(), "frontend")),
    name="static",
)

UPLOAD_DIR = os.path.join(os.getcwd(), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)


def ensure_db():
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM USERS")
        _ = cur.fetchone()
        cur.close()
        conn.close()
        print("База данных доступна, используем существующую.")
    except Exception as e:
        print(f"База данных не доступна или таблицы отсутствуют: {e}")
        print("Инициализация базы данных...")
        asyncio.subprocess.run(
            ["python", os.path.join("backend", "database", "drop_db.py")]
        )
        asyncio.subprocess.run(
            ["python", os.path.join("backend", "database", "init_db.py")]
        )
        print("База данных инициализирована.")


ensure_db()


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/web/process", response_class=HTMLResponse)
async def web_process(request: Request, file: UploadFile = File(...)):
    image_bytes = await file.read()

    loop = asyncio.get_event_loop()
    processed_image, api_data = await loop.run_in_executor(
        None, process_image, image_bytes
    )

    img_base64 = base64.b64encode(processed_image).decode()

    return templates.TemplateResponse(
        "result.html",
        {
            "request": request,
            "image_base64": img_base64,
            "faces": api_data["faces"],
        },
    )


@app.post("/api/process")
async def api_process(file: UploadFile = File(...)):
    image_bytes = await file.read()

    loop = asyncio.get_event_loop()
    processed_image, api_data = await loop.run_in_executor(
        None, process_image, image_bytes
    )

    connection = get_connection()
    cursor = connection.cursor()

    img_uuid = str(uuid.uuid4())
    file_path = os.path.join(UPLOAD_DIR, f"{img_uuid}.jpg")
    with open(file_path, "wb") as f:
        f.write(image_bytes)

    cursor.execute(
        """
        INSERT INTO UPLOADED_IMAGES (UUID, USER_ID, IMAGE_URL, ORIGINAL_FILENAME)
        VALUES (:1, NULL, :2, :3)
        """,
        (img_uuid, file_path, file.filename),
    )

    for face in api_data["faces"]:
        bbox_json = json.dumps(face["bbox"])
        human_uuid = face["identity"]
        identity_conf = face.get("identity_confidence", 0)

        det_uuid = str(uuid.uuid4())
        cursor.execute(
            """
            INSERT INTO FACE_DETECTIONS (
                UUID,
                UPLOADED_IMAGE_ID,
                DETECTED_HUMAN_ID,
                SOURCE_PHOTO_ID,
                DETECTED_BBOX,
                CONFIDENCE,
                EMOTION_CODE
            )
            VALUES (:1, :2, :3, NULL, :4, :5, :6)
            """,
            (
                det_uuid,
                img_uuid,
                human_uuid,
                bbox_json,
                float(identity_conf),
                face["emotion"],
            ),
        )

    connection.commit()
    cursor.close()
    connection.close()

    return JSONResponse(api_data)
