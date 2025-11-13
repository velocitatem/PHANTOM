import { NextRequest, NextResponse } from 'next/server';
import { stopExperimentById, getExperiment } from '@/lib/sessionStore';

export async function POST(req: NextRequest) {
  try {
    const body = await req.json();
    const { experimentId } = body;

    if (!experimentId) {
      return NextResponse.json(
        { error: 'experimentId required' },
        { status: 400 }
      );
    }

    // verify experiment exists
    const existing = getExperiment(experimentId);
    if (!existing) {
      return NextResponse.json(
        { error: 'experiment not found' },
        { status: 404 }
      );
    }

    // stop the experiment
    const exp = stopExperimentById(experimentId);

    return NextResponse.json({
      experimentId: exp!.id,
      status: exp!.status,
    });
  } catch (err: any) {
    console.error('experiment stop error:', err);
    return NextResponse.json(
      { error: err.message || 'unknown error' },
      { status: 500 }
    );
  }
}
