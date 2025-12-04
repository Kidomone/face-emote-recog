from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
import asyncio
import time
from io import BytesIO
import base64

from ML.process_image_file import process_image


app = FastAPI()
templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Web interface for image upload"""
    html = """
    <!DOCTYPE html>
    <html>
    <head><title>Image Processor</title></head>
    <body>
        <h1>Upload Image</h1>
        <form action="/web/process" method="post" enctype="multipart/form-data">
            <input type="file" name="file" accept="image/*" required>
            <button type="submit">Process</button>
        </form>
    </body>
    </html>
    """
    return HTMLResponse(content=html)

@app.post("/web/process", response_class=HTMLResponse)
async def web_process(file: UploadFile = File(...)):
    """Web endpoint - returns HTML with processed image"""
    image_bytes = await file.read()
    
    # Run blocking process_image in thread pool
    loop = asyncio.get_event_loop()
    processed_image, api_data = await loop.run_in_executor(
        None, process_image, image_bytes
    )
    
    # Convert to base64 for display
    img_base64 = base64.b64encode(processed_image).decode()
    
    html = f"""
    <!DOCTYPE html>
    <html>
    <head><title>Processed Image</title></head>
    <body>
        <h1>Processed Image</h1>
        <img src="data:image/jpeg;base64,{img_base64}" style="max-width: 800px;">
        <br><a href="/">Process Another</a>
    </body>
    </html>
    """
    return HTMLResponse(content=html)

@app.post("/api/process")
async def api_process(file: UploadFile = File(...)):
    """API endpoint - returns only JSON data"""
    image_bytes = await file.read()
    
    # Run blocking process_image in thread pool
    loop = asyncio.get_event_loop()
    _, api_data = await loop.run_in_executor(
        None, process_image, image_bytes
    )
    
    return JSONResponse(content=api_data)