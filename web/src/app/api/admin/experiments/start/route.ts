import { NextRequest, NextResponse } from 'next/server';
import { randomUUID } from 'crypto';
import { createExperiment, getSession } from '@/lib/sessionStore';

export async function POST(req: NextRequest) {
  try {
    const body = await req.json();
    const { sessionId } = body;

    if (!sessionId) {
      return NextResponse.json(
        { error: 'sessionId required' },
        { status: 400 }
      );
    }

    // verify session exists
    const session = getSession(sessionId);
    if (!session) {
      return NextResponse.json(
        { error: 'session not found' },
        { status: 404 }
      );
    }

    // generate and create experiment
    const experimentId = randomUUID();
    const exp = createExperiment(sessionId, experimentId);

    return NextResponse.json({
      experimentId: exp.id,
      sessionId,
      status: exp.status,
      createdAt: exp.createdAt,
    });
  } catch (err: any) {
    console.error('experiment start error:', err);
    return NextResponse.json(
      { error: err.message || 'unknown error' },
      { status: 500 }
    );
  }
}
