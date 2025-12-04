FROM python:3.11-slim

WORKDIR /app

# Install system dependencies including graphviz
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    graphviz \
    libgraphviz-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY backend/provider/requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Structure will be mounted via volumes:
# /app/lib -> lib/
# /app/procesing -> experiments/procesing/
# /app/provider -> backend/provider/

ENV PYTHONPATH=/app:/app/lib:/app/procesing

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "5001", "--app-dir", "/app/provider"]
