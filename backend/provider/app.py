from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Any, Literal
import uvicorn
import os
import sys
import json
import numpy as np
import pandas as pd
from supabase import create_client, Client

try:
    # in docker container, paths are set via PYTHONPATH
    from model_registry import ModelRegistry
    from pricing import StateSpace, ElasticityBasedPricingFunction
except ImportError:
    # local dev: add paths manually
    lib_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../lib'))
    procesing_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../experiments/procesing'))
    sys.path.insert(0, lib_path)
    sys.path.insert(0, procesing_path)

    from model_registry import ModelRegistry
    from pricing import StateSpace, ElasticityBasedPricingFunction

SUPABASE_URL = os.getenv("NEXT_PUBLIC_SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("NEXT_PUBLIC_SUPABASE_ANON_KEY", "")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

app = FastAPI(title="PHANTOM Pricing Provider")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

registry = ModelRegistry()

class PriceResponse(BaseModel):
    productId: str
    price: float
    base_price: float
    markup: float
    elasticity: Optional[float] = None
    model_version: str = 'latest'

@app.get("/health")
def health():
    redis_ok = registry.health_check()
    return {"status": "healthy", "redis": redis_ok}

@app.get("/api/{mode}/price/{productId}")
def get_price(
    mode: Literal['hotel', 'airline'],
    productId: str,
    sessionId: Optional[str] = Query(None),
    experimentId: Optional[str] = Query(None)
) -> PriceResponse:
    """
    Dynamic pricing endpoint.

    1. Fetch product base price from Supabase
    2. Load latest elasticity + pricing model from registry
    3. Compute optimal price
    5. Return price to client
    """
    # fetch product
    product_resp = supabase.table(f'{mode}_products').select("id, metadata").eq('id', productId).execute()

    if not product_resp.data or len(product_resp.data) == 0:
        raise HTTPException(status_code=404, detail=f"Product {productId} not found")

    product = product_resp.data[0]
    metadata = product.get('metadata', {})
    base_price = metadata.get('base_price', 100.0)

    # load elasticity data
    elasticity_df = registry.get_elasticity('latest')

    if elasticity_df is None:
        # no model available, return base price
        final_price = base_price
        elasticity_value = None
    else:
        # load or create pricing model
        pricing_model = registry.get_pricing_model('latest')

        if pricing_model is None:
            # create default model
            pricing_model = ElasticityBasedPricingFunction()
            pricing_model.fit(elasticity_df)

        # get elasticity for this product
        product_elasticity = elasticity_df[elasticity_df['productId'] == productId]
        elasticity_value = float(product_elasticity['elasticity'].iloc[0]) if not product_elasticity.empty else None

        # construct state space (single product)
        state = StateSpace(
            demand=np.array([0.0]),  # demand not needed for elasticity-based pricing
            prices=np.array([base_price]),
            session_features=pd.DataFrame()
        )

        # compute optimal price
        optimal_prices = pricing_model.transform(state, product_ids=np.array([productId]))
        final_price = float(optimal_prices[0])

    return PriceResponse(
        productId=productId,
        price=final_price,
        base_price=base_price,
        markup=final_price / base_price if base_price > 0 else 1.0,
        elasticity=elasticity_value,
        model_version='latest'
    )

@app.get("/models")
def list_models():
    """List all registered models in the registry."""
    return registry.list_models()

@app.post("/models/reload")
def reload_models():
    """Force reload of models from registry (useful after pipeline updates)."""
    elasticity = registry.get_elasticity('latest')
    pricing_model = registry.get_pricing_model('latest')

    return {
        "elasticity_loaded": elasticity is not None,
        "n_products": len(elasticity) if elasticity is not None else 0,
        "pricing_model_loaded": pricing_model is not None,
        "model_class": pricing_model.__class__.__name__ if pricing_model else None
    }

if __name__ == "__main__":
    port = int(os.getenv("PROVIDER_PORT", "5001"))
    uvicorn.run(app, host="0.0.0.0", port=port)
