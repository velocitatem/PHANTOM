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

# Copy application code into image
COPY lib/ /app/lib/
COPY experiments/procesing/ /app/procesing/
COPY backend/provider/ /app/provider/

ENV PYTHONPATH=/app:/app/lib:/app/procesing

WORKDIR /app/provider

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "5001"]
