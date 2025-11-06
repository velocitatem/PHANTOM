import { NextRequest, NextResponse } from 'next/server';
import { randomUUID } from 'crypto';

const COOKIE_NAME = 'phantom_session_id';
const isProd = process.env.NODE_ENV === 'production';

export async function GET(req: NextRequest) {
    try {
        // check for existing session cookie
        const existingSession = req.cookies.get(COOKIE_NAME)?.value;

        if (existingSession) {
            return NextResponse.json({ sessionId: existingSession });
        }

        // mint new session id
        const sessionId = randomUUID();

        const res = NextResponse.json({ sessionId });

        // set httpOnly cookie with security flags
        res.cookies.set({
            name: COOKIE_NAME,
            value: sessionId,
            httpOnly: true,
            sameSite: 'lax',
            secure: isProd,
            path: '/',
            maxAge: 60 * 60 * 24 * 30, // 30 days
        });

        return res;
    } catch (err: any) {
        console.error('session error:', err);
        return NextResponse.json(
            { error: err.message || 'unknown error' },
            { status: 500 }
        );
    }
}
