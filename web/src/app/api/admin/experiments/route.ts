import { NextResponse } from 'next/server';
import { getAllExperiments } from '@/lib/sessionStore';

export async function GET() {
  try {
    const exps = getAllExperiments();
    return NextResponse.json({ experiments: exps });
  } catch (err: any) {
    console.error('experiments list error:', err);
    return NextResponse.json(
      { error: err.message || 'unknown error' },
      { status: 500 }
    );
  }
}
