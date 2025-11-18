import { NextRequest, NextResponse } from 'next/server';
import { randomUUID } from 'crypto';
import { getSession, createSession, setExperiment } from '@/lib/sessionStore';

const COOKIE_NAME = 'phantom_session_id';
const isProd = process.env.NODE_ENV === 'production';

export async function GET(req: NextRequest) {
    try {
        const existingSession = req.cookies.get(COOKIE_NAME)?.value;

        if (existingSession) {
            const sessionData = getSession(existingSession);
            return NextResponse.json({
                sessionId: existingSession,
                experimentId: sessionData?.experimentId,
            });
        }

        const sessionId = randomUUID();
        createSession(sessionId);

        const res = NextResponse.json({ sessionId, experimentId: undefined });

        res.cookies.set({
            name: COOKIE_NAME,
            value: sessionId,
            httpOnly: true,
            sameSite: 'lax',
            secure: isProd,
            path: '/',
            maxAge: 60 * 60 * 24 * 30,
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

export async function POST(req: NextRequest) {
    try {
        const body = await req.json();
        const { experimentId } = body;

        if (!experimentId) {
            return NextResponse.json(
                { error: 'experimentId is required' },
                { status: 400 }
            );
        }

        let sessionId = req.cookies.get(COOKIE_NAME)?.value;

        if (!sessionId) {
            sessionId = randomUUID();
            createSession(sessionId);
        }

        setExperiment(sessionId, experimentId);

        const res = NextResponse.json({
            sessionId,
            experimentId,
            success: true
        });

        if (!req.cookies.get(COOKIE_NAME)) {
            res.cookies.set({
                name: COOKIE_NAME,
                value: sessionId,
                httpOnly: true,
                sameSite: 'lax',
                secure: isProd,
                path: '/',
                maxAge: 60 * 60 * 24 * 30,
            });
        }

        return res;
    } catch (err: any) {
        console.error('session update error:', err);
        return NextResponse.json(
            { error: err.message || 'unknown error' },
            { status: 500 }
        );
    }
}
