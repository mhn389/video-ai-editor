FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements-core.txt ./
RUN pip install --no-cache-dir -r requirements-core.txt
# Optional (AGPL-3.0): uncomment if you want YOLO inside the image
# COPY requirements-yolo.txt ./
# RUN pip install --no-cache-dir -r requirements-yolo.txt
COPY features.config.json ./
COPY src ./src
CMD ["python", "src/demo_multilang_info_cpu.py", "--help"]
