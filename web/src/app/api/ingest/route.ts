import { NextRequest, NextResponse } from 'next/server';
import { sendEvent } from '@/lib/kafka';
import type { EventBase } from '@/lib/events';

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

    await sendEvent(event);

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
