import os
from dataclasses import dataclass
@dataclass(frozen=True)
class ServiceConfig:
    web_url: str
    backend_url: str
    pricing_url: str
    kafka_host: str
    kafka_port: int
    redis_port: int
@dataclass(frozen=True)
class TestConfig:
    headless: bool
    timeout: int
    poll_interval: float
    max_retries: int
    kafka_consumer_timeout: int


def load_service_config() -> ServiceConfig:
    return ServiceConfig(
        web_url=os.getenv("WEB_URL", "http://localhost:3000"),
        backend_url=os.getenv("BACKEND_URL", "http://localhost:5000"),
        pricing_url=os.getenv("PRICING_PROVIDER_URL", "http://localhost:5001"),
        kafka_host=os.getenv("KAFKA_HOST", "localhost"),
        kafka_port=int(os.getenv("KAFKA_PORT", "9092")),
        redis_port=int(os.getenv("REDIS_PORT", "6377")),
    )
def load_test_config() -> TestConfig:
    return TestConfig(
        headless=os.getenv("HEADLESS", "true").lower() == "true",
        timeout=int(os.getenv("TEST_TIMEOUT", "30000")),
        poll_interval=float(os.getenv("POLL_INTERVAL", "0.5")),
        max_retries=int(os.getenv("MAX_RETRIES", "10")),
        kafka_consumer_timeout=int(os.getenv("KAFKA_TIMEOUT", "15")),
    )
