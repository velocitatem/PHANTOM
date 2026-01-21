from kafka import KafkaConsumer
import json
import os
from dotenv import load_dotenv
load_dotenv()

def get_interactions(
    topic='user-interactions',
    bootstrap_servers=None,
    from_beginning=True,
    max_records=None,
    timeout_ms=5000
):
    """Consume interaction events from Kafka.

    Args:
        topic: Kafka topic name
        bootstrap_servers: Kafka broker address (default from env)
        from_beginning: Start from earliest offset if True
        max_records: Max number of records to fetch (None = all available)
        timeout_ms: Consumer poll timeout

    Returns:
        List of parsed interaction event dicts
    """
    if not bootstrap_servers:
        host = os.getenv('KAFKA_HOST', 'localhost')
        port = os.getenv('KAFKA_PORT', '9092')
        bootstrap_servers = f'{host}:{port}'

    consumer = KafkaConsumer(
        topic,
        bootstrap_servers=bootstrap_servers,
        auto_offset_reset='earliest' if from_beginning else 'latest',
        enable_auto_commit=False,
        value_deserializer=lambda m: json.loads(m.decode('utf-8')),
        consumer_timeout_ms=timeout_ms
    )

    events = []
    try:
        for msg in consumer:
            events.append(msg.value)
            if max_records and len(events) >= max_records:
                break
    finally:
        consumer.close()

    return events

if __name__ == '__main__':
    interactions = get_interactions(max_records=10)
    for event in interactions:
        print(event)
