import { NextRequest, NextResponse } from 'next/server';

export async function GET(req: NextRequest) {
  const { searchParams } = new URL(req.url);
  const type = searchParams.get('type');

  if (!type || !['hotel', 'airline'].includes(type)) {
    return NextResponse.json(
      { error: 'type parameter must be "hotel" or "airline"' },
      { status: 400 }
    );
  }

  try {
    const backendUrl = process.env.BACKEND_URL || 'http://localhost:5000';
    const url = new URL(`${backendUrl}/api/products/${type}`);

    // forward date index offset to backend
    const dateIndex = searchParams.get('dateIndex');
    if (dateIndex) url.searchParams.set('dateIndex', dateIndex);

    const res = await fetch(url.toString());

    if (!res.ok) {
      throw new Error(`Backend returned ${res.status}`);
    }

    const data = await res.json();
    return NextResponse.json(data);
  } catch (error) {
    console.error('[PRODUCTS_PROXY_ERROR]', error);
    return NextResponse.json(
      { error: 'Failed to fetch products' },
      { status: 500 }
    );
  }
}
