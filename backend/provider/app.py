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
    StateSpace,
    PredictPricesStep
)
from procesing import PipelineContext
sys.path.append(os.path.dirname(os.path.abspath(__file__))+ "/../../lib/")
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
    product = supabase.table(f'{mode}_products').select("metadata").eq('id', productId).execute().data[0]
    if not product: raise HTTPException(404, f"Product {productId} not found")

    metadata = product['metadata']
    base_price = metadata.get('base_price', 100.0)

    class Provider(SupabaseProvider, BackendAPIProvider):
        def __init__(self, backend_url: str):
            SupabaseProvider.__init__(self)
            BackendAPIProvider.__init__(self, backend_url=backend_url)

    context = PipelineContext(
        provider=Provider(backend_url=os.getenv("BACKEND_URL")),
        store_mode=mode
    )

    pricing_model = registry.get_pricing_model('latest')
    elasticity_df = registry.get_elasticity('latest')

    if pricing_model is None or elasticity_df is None:
        return PriceResponse(
            productId=productId,
            price=base_price,
            base_price=base_price,
            markup=1.0,
            elasticity=None
        )

    products = context.products
    if products.empty:
        raise HTTPException(500, "No products available in catalog")

    # merge elasticity with product base prices
    products_with_meta = products.copy()
    products_with_meta['base_price'] = products_with_meta['metadata'].apply(
        lambda m: m.get('base_price', 100.0) if isinstance(m, dict) else 100.0
    )

    merged = products_with_meta[['id', 'base_price']].rename(
        columns={'id': 'productId'}
    ).merge(
        elasticity_df[['productId', 'elasticity']],
        on='productId',
        how='left'
    ).fillna({'elasticity': 0.0})

    # compute demand: use pricer's mean_demand if available, else default
    demand_values = (pricing_model.mean_demand
                    if hasattr(pricing_model, 'mean_demand') and pricing_model.mean_demand is not None
                    else np.ones(len(merged)) * 10.0)

    # build state space with session features if sessionId provided
    session_features = pd.DataFrame()
    if sessionId:
        try:
            # fetch recent session interactions from backend
            from procesing.steps.session import ExtractSessionFeaturesStep
            import requests
            from datetime import datetime, timedelta

            t_end = datetime.utcnow()
            t_start = t_end - timedelta(hours=1)
            backend_url = os.getenv("BACKEND_URL")
            print(backend_url)

            resp = requests.get(
                f"{os.getenv('BACKEND_URL')}/api/kafka/dump", # TODO: THIS IS SHIT, must fix this
                params={'topic': 'user-interactions', 't_start': t_start.isoformat(), 't_end': t_end.isoformat()},
                timeout=2
            )

            if resp.ok:
                msgs = resp.json().get('messages', [])
                interactions_df = pd.DataFrame(msgs)

                if not interactions_df.empty and 'sessionId' in interactions_df.columns:
                    session_interactions = interactions_df[interactions_df['sessionId'] == sessionId]

                    if not session_interactions.empty:
                        extractor = ExtractSessionFeaturesStep(context=context)
                        session_features_df = extractor.transform(session_interactions)

                        if not session_features_df.empty:
                            session_features = session_features_df.drop(columns=['sessionId'])
        except Exception as e:
            print(f"[session-features-error] {e}")
            # continue without session features

    state = StateSpace(
        demand=demand_values,
        prices=merged['base_price'].values,
        session_features=session_features,
        product_ids=merged['productId'].values,
        elasticity=merged['elasticity'].values,
        metadata={'sessionId': sessionId, 'experimentId': experimentId}
    )

    oracle = PredictPricesStep(context=context)
    prices_df = oracle.transform((pricing_model, state))

    product_price_row = prices_df[prices_df['productId'] == productId]
    if product_price_row.empty:
        raise HTTPException(404, f"No pricing available for product {productId}")

    optimal_price = float(product_price_row['predicted_price'].iloc[0])

    product_elasticity_row = elasticity_df[elasticity_df['productId'] == productId]
    product_elasticity = (float(product_elasticity_row['elasticity'].iloc[0])
                         if not product_elasticity_row.empty else None)

    return PriceResponse(
        productId=productId,
        price=optimal_price,
        base_price=base_price,
        markup=optimal_price/base_price,
        elasticity=product_elasticity
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
