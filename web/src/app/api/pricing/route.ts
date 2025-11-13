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

    // log in dev
    if (process.env.NODE_ENV === 'development') {
        console.log('[pricing-api]', {
            productId,
            sessionId,
            experimentId,
            storeMode,
            timestamp: new Date().toISOString(),
        });
    }

    if (!productId) {
        return NextResponse.json(
            { error: 'productId is required' },
            { status: 400 }
        );
    }

    // stub: call external pricing provider (random for now)
    const basePrice = 100 + Math.random() * 900; // 100-1000 range
    const price = Math.round(basePrice * 100) / 100;

    const response: PricingResponse = {
        price,
        currency: 'EUR',
        cachedAt: new Date().toISOString(),
    };

    return NextResponse.json(response);
}
