# boilerplate code
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Any
import uvicorn
import os
import json
from datetime import datetime
from kafka import KafkaProducer
from dotenv import load_dotenv
load_dotenv()

app = FastAPI()

# kafka producer - lazy init
_producer: Optional[KafkaProducer] = None

def get_producer() -> KafkaProducer:
    global _producer
    if _producer is None:
        host = os.getenv('KAFKA_HOST', 'localhost')
        port = os.getenv('KAFKA_PORT', '9092')
        broker = f'{host}:{port}' if port else host
        print(f"[KAFKA_INIT] Connecting to broker: {broker}")
        _producer = KafkaProducer(
            bootstrap_servers=[broker],
            value_serializer=lambda v: json.dumps(v).encode('utf-8'),
            key_serializer=lambda k: k.encode('utf-8') if k else None,
            acks=1,
            retries=3,
            max_in_flight_requests_per_connection=5,
            request_timeout_ms=30000,
            api_version_auto_timeout_ms=10000,
        )
        print(f"[KAFKA_INIT] Producer created successfully")
    return _producer

class EventPayload(BaseModel):
    sessionId: str
    eventName: str
    page: str
    productId: Optional[str] = None
    metadata: Optional[dict[str, Any]] = None
    storeMode: str
    userAgent: Optional[str] = None
    ts: Optional[str] = None

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health():
    kafka_status = "unknown"
    try:
        producer = get_producer()
        # attempt to get cluster metadata to verify connection
        producer.bootstrap_connected()
        kafka_status = "connected"
    except Exception as e:
        kafka_status = f"error: {str(e)}"

    return {
        "status": "healthy",
        "kafka": kafka_status,
        "kafka_broker": f"{os.getenv('KAFKA_HOST', 'localhost')}:{os.getenv('KAFKA_PORT', '9092')}"
    }


@app.post("/api/kafka/ingest")
async def ingest_logs(event: EventPayload):
    try:
        if not event.ts:
            event.ts = datetime.utcnow().isoformat() + 'Z'

        producer = get_producer()
        future = producer.send(
            'user-interactions',
            key=event.sessionId,
            value=event.model_dump()
        )
        # add callback for error logging but don't block
        future.add_errback(lambda e: print(f"[KAFKA_SEND_ERROR] {e}"))

        return {"success": True}
    except Exception as e:
        import traceback
        print(f"[ERROR] {e}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/kafka/dump")
def dump_logs():
    # TODO: implement a dump of logs of time period t_start to t_end (params of get)
    # OR: allow for params of last_n logs as a param - creating two modes of the dumping
    pass



if __name__ == "__main__":
    PORT=int(os.getenv("BACKEND_PORT", 5000))
    uvicorn.run("server:app", host="0.0.0.0", port=PORT, reload=True)
