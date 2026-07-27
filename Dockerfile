# DroneGym web app — FastAPI + static canvas frontend.
# Runs on any torch-capable container host (Render, Railway, Fly.io, HF Spaces).
FROM python:3.12-slim

WORKDIR /app

# deps first for layer caching
COPY requirements-web.txt .
RUN pip install --no-cache-dir -r requirements-web.txt

# app code + committed model/replays (assets/) + frontend (web/)
COPY dronegym/ dronegym/
COPY configs/ configs/
COPY assets/ assets/
COPY web/ web/
COPY server.py .

ENV PORT=8000
EXPOSE 8000
CMD ["sh", "-c", "uvicorn server:app --host 0.0.0.0 --port ${PORT:-8000}"]
