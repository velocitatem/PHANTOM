from typing import Any, Dict
import pandas as pd
from .providers.base import DataProvider

class PipelineContext:
    """
    Context for pipeline execution holding config, provider, and cached data.
    Enables dependency injection and eliminates global state.
    """

    def __init__(self,
                 provider: DataProvider,
                 store_mode: str,
                 window_size: str = '30s',
                 **config):
        self.provider = provider
        self.store_mode = store_mode
        self.window_size = window_size
        self.config = config
        self._cache: Dict[str, Any] = {}

    def get_cached(self, key: str, default=None):
        return self._cache.get(key, default)

    def cache(self, key: str, value):
        self._cache[key] = value
        return value

    @property
    def products(self) -> pd.DataFrame:
        """Lazy-load and cache product catalog, single fetch per pipeline run"""
        if 'products' not in self._cache:
            self._cache['products'] = self.provider.fetch_products(self.store_mode)
        return self._cache['products']
