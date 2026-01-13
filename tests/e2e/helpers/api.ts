interface PriceResponse {
  price: number;
  base_price: number;
  markup: number;
  model_version?: string;
}

export async function fetchPrice(
  baseUrl: string,
  productId: string,
  mode: string = 'simple_surge',
  sessionId?: string
): Promise<PriceResponse> {
  const params = new URLSearchParams();
  if (sessionId) params.set('sessionId', sessionId);

  const url = `${baseUrl}/api/pricing?mode=${mode}&productId=${productId}&${params}`;
  const resp = await fetch(url);

  if (!resp.ok) {
    throw new Error(`Price fetch failed: ${resp.status}`);
  }

  return resp.json();
}

export async function waitForPriceChange(
  baseUrl: string,
  productId: string,
  baselinePrice: number,
  mode: string,
  sessionId?: string,
  maxRetries: number = 10,
  pollInterval: number = 500
): Promise<PriceResponse> {
  for (let i = 0; i < maxRetries; i++) {
    const priceResp = await fetchPrice(baseUrl, productId, mode, sessionId);
    if (Math.abs(priceResp.price - baselinePrice) > 0.01) {
      return priceResp;
    }
    await new Promise(r => setTimeout(r, pollInterval));
  }

  throw new Error(`Price did not change after ${maxRetries} retries`);
}

export async function ingestEvent(
  baseUrl: string,
  sessionId: string,
  event: string,
  productId?: string,
  metadata?: Record<string, any>
): Promise<void> {
  const resp = await fetch(`${baseUrl}/api/ingest`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      sessionId,
      event,
      productId,
      timestamp: new Date().toISOString(),
      metadata,
    }),
  });

  if (!resp.ok) {
    throw new Error(`Event ingest failed: ${resp.status}`);
  }
}
