import { NextRequest, NextResponse } from 'next/server';
import { sendInteractionEvent } from '@/lib/kafka';

export async function POST(req: NextRequest) {
    try {
        const body = await req.json();
        const { sessionId, eventType, targetEl, targetUrl, metadata } = body;

        if (!sessionId || !eventType) {
            return NextResponse.json(
                { error: 'sessionId and eventType required' },
                { status: 400 }
            );
        }

        await sendInteractionEvent({
            sessionId,
            eventType,
            targetEl,
            targetUrl,
            metadata,
            ts: Date.now(),
        });

        return NextResponse.json({ success: true });
    } catch (err: any) {
        console.error('track error:', err);
        return NextResponse.json(
            { error: err.message || 'unknown error' },
            { status: 500 }
        );
    }
}
