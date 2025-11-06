import { NextRequest, NextResponse } from 'next/server';
import { randomUUID } from 'crypto';
import { getSession, createSession } from '@/lib/sessionStore';

const COOKIE_NAME = 'phantom_session_id';
const isProd = process.env.NODE_ENV === 'production';

export async function GET(req: NextRequest) {
    try {
        // check for existing session cookie
        const existingSession = req.cookies.get(COOKIE_NAME)?.value;

        if (existingSession) {
            const sessionData = getSession(existingSession);
            return NextResponse.json({
                sessionId: existingSession,
                experimentId: sessionData?.experimentId,
            });
        }

        // mint new session id
        const sessionId = randomUUID();
        createSession(sessionId);

        const res = NextResponse.json({ sessionId, experimentId: undefined });

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
