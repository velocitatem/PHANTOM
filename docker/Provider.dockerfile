FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml /app/
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    pydantic \
    numpy \
    pandas \
    scikit-learn \
    redis \
    supabase \
    kafka \
    confluent-kafka


# structure will be mounted via volumes:
# /app/lib -> lib/
# /app/procesing -> experiments/procesing/
# /app/provider -> backend/provider/

ENV PYTHONPATH=/app:/app/lib:/app/procesing

CMD ["python", "-m", "uvicorn", "provider.app:app", "--host", "0.0.0.0", "--port", "5001"]
