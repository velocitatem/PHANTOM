import { NextRequest, NextResponse } from 'next/server';

export async function GET(
  req: NextRequest,
  { params }: { params: Promise<{ id: string }> }
) {
  const { id } = await params;

  if (!id) {
    return NextResponse.json(
      { error: 'product id is required' },
      { status: 400 }
    );
  }

  try {
    const backendUrl = process.env.BACKEND_URL || 'http://localhost:5000';
    const url = new URL(`${backendUrl}/api/products/${id}`);

    const res = await fetch(url.toString());

    if (!res.ok) {
      throw new Error(`Backend returned ${res.status}`);
    }

    const data = await res.json();
    return NextResponse.json(data);
  } catch (error) {
    console.error('[PRODUCT_DETAIL_ERROR]', error);
    return NextResponse.json(
      { error: 'Failed to fetch product details' },
      { status: 500 }
    );
  }
}
