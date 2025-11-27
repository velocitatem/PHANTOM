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
    const storeMode = process.env.NEXT_PUBLIC_STORE_MODE || 'shop';

    if (!productId) {
        return NextResponse.json(
            { error: 'productId is required' },
            { status: 400 }
        );
    }

    // stub: call external pricing provider (random for now)
    const basePrice = 100 + Math.random() * 900; // 100-1000 range
    const price = Math.round(basePrice * 100) / 100;
    const timestamp = new Date().toISOString();

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
            // don't fail the pricing request if logging fails
        }
    }

    // log in dev
    if (process.env.NODE_ENV === 'development') {
        console.log('[pricing-api]', {
            productId,
            sessionId,
            experimentId,
            storeMode,
            price,
            timestamp,
        });
    }

    const response: PricingResponse = {
        price,
        currency: 'EUR',
        cachedAt: timestamp,
    };

    return NextResponse.json(response);
}
