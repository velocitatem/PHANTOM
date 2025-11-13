import { NextRequest, NextResponse } from 'next/server';
import type { EventBase } from '@/lib/events';

const BACKEND_URL = process.env.BACKEND_URL || 'http://localhost:5000';

export async function POST(req: NextRequest) {
  try {
    const body = await req.json();

    const storeMode = process.env.STORE_MODE || 'hotel';
    const userAgent = req.headers.get('user-agent') || undefined;

    const event: EventBase = {
      ...body,
      storeMode,
      userAgent,
      ts: body.ts || new Date().toISOString(),
    };

    const res = await fetch(`${BACKEND_URL}/api/kafka/ingest`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(event),
    });

    if (!res.ok) {
      throw new Error(`Backend returned ${res.status}`);
    }

    if (process.env.NEXT_PUBLIC_APP_ENV === 'dev') {
      console.log('[ingest]', event);
    }

    return NextResponse.json({ success: true });
  } catch (err: any) {
    console.error('[ingest error]', err);
    return NextResponse.json(
      { error: err.message || 'unknown error' },
      { status: 500 }
    );
  }
}
