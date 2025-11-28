import os
import pandas as pd
import requests
from typing import List
from procesing.providers.base import DataProvider

class BackendAPIProvider(DataProvider):
    """Concrete backend API implementation"""
    def __init__(self, backend_url: str = None):
        self.backend_url = backend_url or os.getenv("BACKEND_URL", "http://localhost:5000")
    def fetch_kafka_topic(self, topic: str) -> pd.DataFrame:
        resp = requests.get(f"{self.backend_url}/api/kafka/dump?topic={topic}")
        resp.raise_for_status()
        data = resp.json()

        if not data.get('success') or not data.get('data'):
            return pd.DataFrame()

        return pd.DataFrame(data['data'])
