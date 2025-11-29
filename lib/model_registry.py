import redis
import pickle
import json
import pandas as pd
from typing import Optional, Dict, Any
import os
import logging
log = logging.getLogger(__name__)

class ModelRegistry:
    """
    Lightweight model registry using Redis for storing pricing models and elasticity data.
    Models are serialized using pickle, metadata stored as JSON.
    """

    def __init__(self, redis_host: str = None, redis_port: int = None):
        host = redis_host or os.getenv('REDIS_HOST', 'localhost')
        port = redis_port or int(os.getenv('REDIS_PORT', '6378'))

        self.redis_client = redis.Redis(
            host=host,
            port=port,
            db=0,
            decode_responses=False
        )
        self.metadata_prefix = "model:meta:"
        self.data_prefix = "model:data:"
        self.elasticity_prefix = "elasticity:"
        self.prices_prefix = "prices:"

    def publish_elasticity(self,
                          elasticity_df: pd.DataFrame,
                          model_name: str = 'latest',
                          metadata: Optional[Dict[str, Any]] = None):
        """
        Store elasticity estimates in registry.

        Args:
            elasticity_df: df with [productId, elasticity, std_error, n_obs]
            model_name: identifier for this elasticity snapshot
            metadata: additional info (timestamp, window_size, etc)
        """
        key = f"{self.elasticity_prefix}{model_name}"

        # serialize dataframe as JSON
        data_json = elasticity_df.to_json(orient='records')

        # store data
        self.redis_client.set(key, data_json)

        # store metadata
        meta = metadata or {}
        meta.update({
            'n_products': len(elasticity_df),
            'mean_elasticity': float(elasticity_df['elasticity'].mean()),
            'model_type': 'elasticity_snapshot'
        })

        meta_key = f"{self.metadata_prefix}{model_name}"
        self.redis_client.set(meta_key, json.dumps(meta))

        log.info(f"Published elasticity model '{model_name}' with {len(elasticity_df)} products")

    def get_elasticity(self, model_name: str = 'latest') -> Optional[pd.DataFrame]:
        """Retrieve elasticity estimates from registry."""
        key = f"{self.elasticity_prefix}{model_name}"
        data_json = self.redis_client.get(key)

        if data_json is None:
            return None

        # decode bytes to string if needed
        if isinstance(data_json, bytes):
            data_json = data_json.decode('utf-8')

        return pd.read_json(data_json, orient='records')

    def publish_pricing_model(self,
                              pricing_function,
                              model_name: str = 'latest',
                              metadata: Optional[Dict[str, Any]] = None):
        """
        Store a fitted pricing function object.

        Args:
            pricing_function: fitted PricingFunction instance
            model_name: identifier
            metadata: additional info
        """
        key = f"{self.data_prefix}{model_name}"

        # serialize object
        model_bytes = pickle.dumps(pricing_function)
        self.redis_client.set(key, model_bytes)

        # store metadata
        meta = metadata or {}
        meta.update({
            'model_class': pricing_function.__class__.__name__,
            'model_type': 'pricing_function'
        })

        meta_key = f"{self.metadata_prefix}{model_name}"
        self.redis_client.set(meta_key, json.dumps(meta))

        log.info(f"Published pricing model '{model_name}' ({meta['model_class']})")

    def get_pricing_model(self, model_name: str = 'latest'):
        """Retrieve a pricing function from registry."""
        key = f"{self.data_prefix}{model_name}"
        model_bytes = self.redis_client.get(key)

        if model_bytes is None:
            return None

        return pickle.loads(model_bytes)

    def list_models(self) -> Dict[str, Any]:
        """List all registered models with metadata."""
        models = {}

        for key in self.redis_client.scan_iter(f"{self.metadata_prefix}*"):
            key_str = key.decode('utf-8') if isinstance(key, bytes) else key
            model_name = key_str.replace(self.metadata_prefix, '')
            meta_json = self.redis_client.get(key)

            if meta_json:
                if isinstance(meta_json, bytes):
                    meta_json = meta_json.decode('utf-8')
                models[model_name] = json.loads(meta_json)

        return models

    def publish_prices(self,
                      prices_df: pd.DataFrame,
                      model_name: str = 'latest',
                      metadata: Optional[Dict[str, Any]] = None):
        """Store predicted prices in registry.

        Args:
            prices_df: df with [productId, predicted_price, ...]
            model_name: identifier for this price snapshot
            metadata: additional info
        """
        key = f"{self.prices_prefix}{model_name}"
        data_json = prices_df.to_json(orient='records')

        self.redis_client.set(key, data_json)

        meta = metadata or {}
        meta.update({
            'n_products': len(prices_df),
            'model_type': 'predicted_prices'
        })

        meta_key = f"{self.metadata_prefix}prices_{model_name}"
        self.redis_client.set(meta_key, json.dumps(meta))

        log.info(f"Published prices '{model_name}' for {len(prices_df)} products")

    def get_prices(self, model_name: str = 'latest') -> Optional[pd.DataFrame]:
        """Retrieve predicted prices from registry."""
        key = f"{self.prices_prefix}{model_name}"
        data_json = self.redis_client.get(key)

        if data_json is None:
            return None

        if isinstance(data_json, bytes):
            data_json = data_json.decode('utf-8')

        return pd.read_json(data_json, orient='records')

    def health_check(self) -> bool:
        """Check if Redis connection is alive."""
        try:
            self.redis_client.ping()
            return True
        except:
            return False
