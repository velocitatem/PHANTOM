from abc import ABC, abstractmethod
from typing import List
import pandas as pd

class DataProvider(ABC):
    """Abstract interface for data access, enables DI and testing"""

    @abstractmethod
    def fetch_products(self, store_mode: str) -> pd.DataFrame:
        """Fetch product catalog for given store mode"""
        pass

    @abstractmethod
    def fetch_experiments(self, experiment_ids: List[str]) -> pd.DataFrame:
        """Fetch experiment metadata for given IDs"""
        pass

    @abstractmethod
    def fetch_kafka_topic(self, topic: str) -> pd.DataFrame:
        """Fetch data from Kafka topic via backend API"""
        pass
