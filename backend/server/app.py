# boilerplate code
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Any
import uvicorn
import os
import json
from datetime import datetime
from kafka import KafkaProducer, KafkaAdminClient, KafkaConsumer
from kafka.admin import NewTopic
from kafka.errors import TopicAlreadyExistsError
from dotenv import load_dotenv
from supabase import create_client, Client
load_dotenv()

app = FastAPI()

# kafka producer - lazy init
_producer: Optional[KafkaProducer] = None

# supabase client - lazy init
_supabase: Optional[Client] = None

def get_supabase() -> Client:
    global _supabase
    if _supabase is None:
        url = os.getenv('NEXT_PUBLIC_SUPABASE_URL')
        key = os.getenv('NEXT_PUBLIC_SUPABASE_ANON_KEY')
        if not url or not key:
            raise ValueError("Supabase credentials not configured")
        _supabase = create_client(url, key)
    return _supabase

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
            max_block_ms=5000,  # don't block send() for more than 5s
        )
        print(f"[KAFKA_INIT] Producer created successfully")
    return _producer

class EventPayload(BaseModel):
    sessionId: str
    experimentId: Optional[str] = None
    eventName: str
    page: str
    productId: Optional[str] = None
    metadata: Optional[dict[str, Any]] = None
    storeMode: str
    userAgent: Optional[str] = None
    ts: Optional[str] = None

class PriceLogPayload(BaseModel):
    productId: str
    price: float
    sessionId: str
    experimentId: Optional[str] = None
    storeMode: str
    ts: Optional[str] = None

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    """create kafka topics on startup"""
    host = os.getenv('KAFKA_HOST', 'localhost')
    port = os.getenv('KAFKA_PORT', '9092')
    broker = f'{host}:{port}'

    try:
        print(f"[STARTUP] Creating Kafka topics on {broker}")
        admin = KafkaAdminClient(
            bootstrap_servers=[broker],
            request_timeout_ms=10000,
        )

        topics = [
            NewTopic(name='user-interactions', num_partitions=3, replication_factor=1),
            NewTopic(name='price-logs', num_partitions=3, replication_factor=1)
        ]

        admin.create_topics(new_topics=topics, validate_only=False)
        print(f"[STARTUP] Topics created successfully")
        admin.close()
    except TopicAlreadyExistsError:
        print(f"[STARTUP] Topics already exist, skipping creation")
    except Exception as e:
        print(f"[STARTUP] Failed to create topics: {e}")
        print(f"[STARTUP] Will rely on auto-creation on first message")

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

@app.post("/api/kafka/price-log")
async def ingest_price_log(price_log: PriceLogPayload):
    try:
        if not price_log.ts:
            price_log.ts = datetime.utcnow().isoformat() + 'Z'

        producer = get_producer()
        future = producer.send(
            'price-logs',
            key=price_log.productId,
            value=price_log.model_dump()
        )
        future.add_errback(lambda e: print(f"[KAFKA_PRICE_LOG_ERROR] {e}"))

        return {"success": True}
    except Exception as e:
        import traceback
        print(f"[PRICE_LOG_ERROR] {e}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/kafka/dump")
def dump_logs(
    topic: str = 'user-interactions',
    last_n: Optional[int] = None,
    t_start: Optional[str] = None,
    t_end: Optional[str] = None
):
    """dump all messages from specified kafka topic

    params:
        topic: kafka topic to dump (default: user-interactions)
        last_n: return only last n messages (default: all)
        t_start: filter by start timestamp iso format
        t_end: filter by end timestamp iso format
    """
    if topic not in ['user-interactions', 'price-logs']:
        raise HTTPException(status_code=400, detail="Invalid topic")

    host = os.getenv('KAFKA_HOST', 'localhost')
    port = os.getenv('KAFKA_PORT', '9092')
    broker = f'{host}:{port}'

    try:
        consumer = KafkaConsumer(
            topic,
            bootstrap_servers=[broker],
            auto_offset_reset='earliest',
            enable_auto_commit=False,
            value_deserializer=lambda x: json.loads(x.decode('utf-8')),
            consumer_timeout_ms=5000
        )

        events = []
        for msg in consumer:
            events.append(msg.value)

        consumer.close()

        # apply filters
        if t_start or t_end:
            filtered = []
            for e in events:
                ts = e.get('ts')
                if ts:
                    if t_start and ts < t_start:
                        continue
                    if t_end and ts > t_end:
                        continue
                filtered.append(e)
            events = filtered

        if last_n and last_n > 0:
            events = events[-last_n:]

        return {"success": True, "count": len(events), "data": events}

    except Exception as e:
        import traceback
        print(f"[DUMP_ERROR] {e}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/products/{product_id}")
async def get_product_by_id(product_id: str):
    """fetch single product by id from either hotel_products or airline_products"""
    try:
        supabase = get_supabase()

        # try hotel_products first
        response = supabase.table('hotel_products').select('*').eq('id', product_id).execute()
        if response.data and len(response.data) > 0:
            return {"success": True, "data": response.data[0]}

        # try airline_products
        response = supabase.table('airline_products').select('*').eq('id', product_id).execute()
        if response.data and len(response.data) > 0:
            return {"success": True, "data": response.data[0]}

        raise HTTPException(status_code=404, detail="Product not found")

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        print(f"[PRODUCT_BY_ID_ERROR] {e}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/products/type/{product_type}")
async def get_products(
    product_type: str,
    dateIndex: Optional[int] = None,
    origin: Optional[str] = None,
    destination: Optional[str] = None,
    tripType: Optional[str] = None,
    adults: Optional[int] = None,
    children: Optional[int] = None,
    infants: Optional[int] = None,
    rooms: Optional[int] = None
):
    """fetch products from supabase based on type (hotel or airline)

    params:
        product_type: either 'hotel' or 'airline'
        dateIndex: optional days offset from today (e.g., 0=today, 1=tomorrow, -1=yesterday)
        origin: (airline) departure airport code
        destination: (airline/hotel) arrival airport or hotel location
        tripType: (airline) roundtrip, oneway, multicity
        adults, children, infants: passenger counts
        rooms: (hotel) number of rooms
    """
    if product_type not in ['hotel', 'airline']:
        raise HTTPException(status_code=400, detail="product_type must be 'hotel' or 'airline'")

    try:
        supabase = get_supabase()
        table = f'{product_type}_products'

        query = supabase.table(table).select('*')

        # filter by exact date_index if provided
        # dateIndex from frontend is days from today, convert to days since epoch
        if dateIndex is not None:
            from datetime import datetime
            epoch = datetime(1970, 1, 1)
            today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
            today_index = int((today - epoch).total_seconds() / 86400)
            actual_date_index = today_index + dateIndex
            query = query.eq('date_index', actual_date_index)

        response = query.execute()
        results = response.data

        # apply in-memory filters based on metadata for airline products
        if product_type == 'airline' and results:
            filtered = []
            for product in results:
                metadata = product.get('metadata', {})

                # filter by origin airport
                if origin:
                    dep = metadata.get('departure', {})
                    if dep.get('airport') != origin:
                        continue

                # filter by destination airport
                if destination:
                    arr = metadata.get('arrival', {})
                    if arr.get('airport') != destination:
                        continue

                # passenger count validation (ensure total capacity)
                if adults is not None or children is not None or infants is not None:
                    total_pax = (adults or 0) + (children or 0) + (infants or 0)
                    avail = product.get('availability', 0)
                    if avail < total_pax:
                        continue

                filtered.append(product)

            results = filtered

        # apply in-memory filters for hotel products
        elif product_type == 'hotel' and results:
            filtered = []
            for product in results:
                metadata = product.get('metadata', {})

                # filter by occupancy capacity
                if adults is not None:
                    max_occ = metadata.get('max_occupancy', 2)
                    if max_occ < adults:
                        continue

                # filter by room availability
                if rooms is not None:
                    avail = product.get('availability', 0)
                    if avail < rooms:
                        continue

                filtered.append(product)

            results = filtered

        return {"success": True, "count": len(results), "data": results}

    except Exception as e:
        import traceback
        print(f"[PRODUCTS_ERROR] {e}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))



if __name__ == "__main__":
    PORT=int(os.getenv("BACKEND_PORT", 5000))
    uvicorn.run("server:app", host="0.0.0.0", port=PORT, reload=True)
