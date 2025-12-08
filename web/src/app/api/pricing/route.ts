import { NextRequest, NextResponse } from 'next/server';

interface PricingResponse {
    price: number;
    currency: string;
    cachedAt: string;
}

export async function GET(req: NextRequest) {
    const { searchParams } = new URL(req.url);
    const productId = searchParams.get('productId');
    const sessionId = searchParams.get('sessionId');
    const experimentId = searchParams.get('experimentId');
    const storeMode = process.env.NEXT_PUBLIC_STORE_MODE || process.env.STORE_MODE || 'hotel';

    if (!productId) {
        return NextResponse.json(
            { error: 'productId is required' },
            { status: 400 }
        );
    }

    const timestamp = new Date().toISOString();
    let price: number;
    let basePrice: number | undefined;
    let markup: number | undefined;
    let elasticity: number | undefined;

    // call real pricing provider
    const providerUrl = process.env.PRICING_PROVIDER_URL || 'http://localhost:5001';
    try {
        const queryParams = new URLSearchParams();
        if (sessionId) queryParams.append('sessionId', sessionId);
        if (experimentId) queryParams.append('experimentId', experimentId);

        const providerResponse = await fetch(
            `${providerUrl}/api/${storeMode}/price/${productId}?${queryParams.toString()}`,
            { headers: { 'Accept': 'application/json' }, cache: 'no-store' }
        );

        if (!providerResponse.ok) {
            throw new Error(`Provider returned ${providerResponse.status}`);
        }

        const providerData = await providerResponse.json();
        price = providerData.price;
        basePrice = providerData.base_price;
        markup = providerData.markup;
        elasticity = providerData.elasticity;

    } catch (err) {
        console.error('[pricing-provider-error]', err);
        // fallback to random pricing if provider unavailable
        const randomBase = 100 + Math.random() * 900;
        price = Math.round(randomBase * 100) / 100;
    }

    // log price to kafka for elasticity computation
    if (sessionId) {
        const backendUrl = process.env.BACKEND_URL || 'http://localhost:5000';
        try {
            await fetch(`${backendUrl}/api/kafka/price-log`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    productId,
                    price,
                    sessionId,
                    experimentId: experimentId || undefined,
                    storeMode,
                    ts: timestamp,
                }),
            });
        } catch (err) {
            console.error('[price-log-error]', err);
        }
    }

    if (process.env.NODE_ENV === 'development') {
        console.log('[pricing-api]', {
            productId, sessionId, experimentId, storeMode,
            price, basePrice, markup, elasticity, timestamp,
        });
    }

    const response: PricingResponse = {
        price,
        currency: 'EUR',
        cachedAt: timestamp,
    };

    return NextResponse.json(response);
}
