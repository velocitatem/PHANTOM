import { useEffect, useRef, useState } from 'react';
import '@/lib/experiments' // ensure experiments lib is loaded

const fetchSessionId = async (): Promise<string> => {
    try {
        const res = await fetch('/api/session');
        const data = await res.json();
        return data.sessionId || '';
    } catch (err) {
        console.error('failed to fetch session:', err);
        return '';
    }
};

const track = async (ev: {
    sessionId: string;
    eventType: string;
    targetEl?: string;
    targetUrl?: string;
    metadata?: Record<string, any>;
}) => {
    try {
        await fetch('/api/track', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(ev),
        });
    } catch (err) {
        console.error('track failed:', err);
    }
};

export const useInteractionTracking = () => {
    const sidRef = useRef<string>('');
    const [ready, setReady] = useState(false);

    useEffect(() => {
        // fetch session id from httpOnly cookie via API
        fetchSessionId().then((sid) => {
            sidRef.current = sid;
            setReady(true);
        });

        const handleClick = (e: MouseEvent) => {
            if (!sidRef.current) return;
            const tgt = e.target as HTMLElement;
            track({
                sessionId: sidRef.current,
                eventType: 'click',
                targetEl: tgt.tagName,
                targetUrl: tgt instanceof HTMLAnchorElement ? tgt.href : undefined,
                metadata: {
                    x: e.clientX,
                    y: e.clientY,
                    path: window.location.pathname,
                },
            });
        };

        const handleScroll = () => {
            if (!sidRef.current) return;
            track({
                sessionId: sidRef.current,
                eventType: 'scroll',
                metadata: {
                    scrollY: window.scrollY,
                    path: window.location.pathname,
                },
            });
        };

        const handlePageView = () => {
            if (!sidRef.current) return;
            track({
                sessionId: sidRef.current,
                eventType: 'pageview',
                metadata: {
                    path: window.location.pathname,
                    referrer: document.referrer,
                },
            });
        };

        enum DefinedInteractions {
            ADD_TO_CART = 'add_to_cart',
            PURCHASE = 'purchase',
        }

        // called when clicking on "Add to Cart" button or "Purchase" button
        const handleDefinedInteraction = (
            interactionType: DefinedInteractions,
            metadata?: Record<string, any>
        ) => {
            if (!sidRef.current) return;
            track({
                sessionId: sidRef.current,
                eventType: interactionType,
                metadata: {
                    path: window.location.pathname,
                    ...metadata,
                },
            });
        };

        const definedInteractionListener = (e: Event) => {
            const customEvent = e as CustomEvent;
            handleDefinedInteraction(customEvent.detail.interactionType, customEvent.detail.metadata);
        };

        // wait for session to be ready before tracking
        if (!ready) return;

        handlePageView();
        document.addEventListener('click', handleClick);
        document.addEventListener('definedInteraction', definedInteractionListener);
        // TOO NOISY: enable if needed but tbh not worth it
        //window.addEventListener('scroll', handleScroll, { passive: true });

        return () => {
            document.removeEventListener('click', handleClick);
            document.removeEventListener('definedInteraction', definedInteractionListener);
            //window.removeEventListener('scroll', handleScroll);
        };
    }, [ready]);
};
