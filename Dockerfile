FROM python:3.11-slim

WORKDIR /app

# Install system deps for opencv
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy only what's needed to run the API
COPY src/ ./src/
COPY api.py .

# Copy checkpoint
COPY mpv2_output/checkpoints/best_model.pth ./mpv2_output/checkpoints/best_model.pth

# Output dir for predictions
RUN mkdir -p predictions/api/masks predictions/api/overlays

EXPOSE 8000

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
