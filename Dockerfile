FROM python:3.12

WORKDIR /app

COPY requirements.txt .
RUN python -m pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

COPY api ./api
COPY ML ./ML

EXPOSE 8000

CMD ["python", "api/main.py"]
