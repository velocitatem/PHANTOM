import { NextRequest, NextResponse } from 'next/server';

export function proxy(req: NextRequest) {
    const mode = process.env.STORE_MODE;
    const { pathname } = req.nextUrl;

    // skip rewrites for api routes, admin routes, static files, and next internals
    if (
        pathname.startsWith('/api') ||
        pathname.startsWith('/admin') ||
        pathname.startsWith('/_next') ||
        pathname.startsWith('/static') ||
        pathname.startsWith('/start-task') ||
        pathname.startsWith('/cart') ||
        pathname.includes('.')
        // TODO: add robots.txt and sitemap.xml if needed here
    ) {
        return NextResponse.next();
    }

    // already prefixed with mode
    if (pathname.startsWith(`/${mode}`)) {
        return NextResponse.next();
    }

    // rewrite root and unprefixed paths to mode-specific route group
    const url = req.nextUrl.clone();
    url.pathname = `/${mode}${pathname === '/' ? '' : pathname}`;

    return NextResponse.rewrite(url);
}

export const config = {
    matcher: [
        // match all paths except those starting with _next/static, _next/image, favicon.ico
        '/((?!_next/static|_next/image|favicon.ico).*)',
    ],
};
