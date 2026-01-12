from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Literal, Optional
import uvicorn, os, sys
from supabase import create_client, Client
from dotenv import load_dotenv
import numpy as np
import pandas as pd
load_dotenv()

# Local imports of registry and pricing function

sys.path.append(os.path.dirname(os.path.abspath(__file__))+ "/../../experiments/")
from procesing.providers import SupabaseProvider, BackendAPIProvider
from procesing.pricers import (
    StaticPricer,
    RandomPricer,
    ElasticityBasedPricer
)
from procesing.steps import (
    PredictPricesStep
)
from procesing import PipelineContext
sys.path.append(os.path.dirname(os.path.abspath(__file__))+ "/../../lib/")
print(os.path.dirname(os.path.abspath(__file__))+ "/../../lib/")
from lib.model_registry import ModelRegistry

# Config
app = FastAPI(title="PHANTOM Pricing Provider")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

supabase: Client = create_client(os.getenv("NEXT_PUBLIC_SUPABASE_URL"), os.getenv("NEXT_PUBLIC_SUPABASE_ANON_KEY"))
registry = ModelRegistry()

class PriceResponse(BaseModel):
    productId: str
    price: float
    base_price: float
    markup: float
    elasticity: Optional[float] = None
    model_version: str = 'latest'

@app.get("/health")
def health() -> dict:
    return {"status": "healthy", "redis": registry.health_check()}

@app.get("/api/{mode}/price/{productId}", response_model=PriceResponse)
def get_price(mode: Literal['hotel', 'airline'], productId: str, sessionId: Optional[str] = Query(None), experimentId: Optional[str] = Query(None)):
    """
    THIS is the fast lookup service (mechanism).
    Priority: session-keyed price > global optimal price > base price
    """
    product = supabase.table(f'{mode}_products').select("metadata").eq('id', productId).execute().data[0]
    if not product: raise HTTPException(404, f"Product {productId} not found")

    metadata = product['metadata']
    base_price = metadata.get('base_price', 100.0)

    # PRIORITY 1: session-aware price (computed by Airflow worker)
    if sessionId:
        session_price = registry.get_session_price(sessionId, productId)
        if session_price is not None:
            return PriceResponse(
                productId=productId,
                price=session_price,
                base_price=base_price,
                markup=session_price/base_price,
                elasticity=None,
                model_version='session-aware'
            )

    # PRIORITY 2: global pre-computed prices (surge pricing)
    prices_df = registry.get_prices('latest')
    if prices_df is not None:
        product_price_row = prices_df[prices_df['productId'] == productId]
        if not product_price_row.empty:
            optimal_price = float(product_price_row['optimal_price'].iloc[0])
            return PriceResponse(
                productId=productId,
                price=optimal_price,
                base_price=base_price,
                markup=optimal_price/base_price,
                elasticity=None,
                model_version='surge'
            )

    # PRIORITY 3: fallback to base price
    return PriceResponse(
        productId=productId,
        price=base_price,
        base_price=base_price,
        markup=1.0,
        elasticity=None,
        model_version='base'
    )

@app.get("/models")
def list_models(): return registry.list_models()

@app.post("/models/reload")
def reload_models():
    elasticity, pricing_model = registry.get_elasticity('latest'), registry.get_pricing_model('latest')
    return {
        "elasticity_loaded": bool(elasticity),
        "n_products": len(elasticity) if elasticity is not None else 0,
        "pricing_model_loaded": bool(pricing_model),
        "model_class": pricing_model.__class__.__name__ if pricing_model else None
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PROVIDER_PORT", "5001")))
