from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Literal, Optional
import uvicorn, os, sys
from supabase import create_client, Client

# Local imports of registry and pricing function


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
    elasticity_df = registry.get_elasticity('latest')

    if not elasticity_df:
        return PriceResponse(productId=productId, price=base_price, base_price=base_price, markup=1.0)

    pricing_model = registry.get_pricing_model('latest') or ElasticityBasedPricingFunction().fit(elasticity_df)
    product_elasticity = elasticity_df[elasticity_df['productId'] == productId]['elasticity'].iloc[0] if (elasticity_row := elasticity_df[elasticity_df['productId'] == productId]).any().any() else None

    state = StateSpace(np.array([0.0]), np.array([base_price]), pd.DataFrame())
    optimal_price = pricing_model.transform(state, np.array([productId]))[0]

    return PriceResponse(
        productId=productId,
        price=float(optimal_price),
        base_price=base_price,
        markup=optimal_price/base_price,
        elasticity=float(product_elasticity) if product_elasticity is not None else None
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
