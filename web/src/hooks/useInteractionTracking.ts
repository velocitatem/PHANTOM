import { useEffect, useRef } from 'react';
import '@/lib/experiments' // ensure experiments lib is loaded

const genSessionId = () => {
    if (typeof window === 'undefined') return '';
    let sid = sessionStorage.getItem('phantom_session_id');
    if (!sid) {
        sid = `${Date.now()}-${Math.random().toString(36).slice(2)}`;
        sessionStorage.setItem('phantom_session_id', sid);
        // TODO: when creating new id send to exepriemtn tracking db
        // match between sesion-id and experiment-id for this session
        // so that we can identify all interactions aligning with a specific experiment goal.
    }
    return sid;
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

    useEffect(() => {
        sidRef.current = genSessionId();

        const handleClick = (e: MouseEvent) => {
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
            track({
                sessionId: sidRef.current,
                eventType: interactionType,
                metadata: {
                    path: window.location.pathname,
                    ...metadata,
                },
            });
        };


        handlePageView();
        document.addEventListener('click', handleClick);
        document.addEventListener('definedInteraction', (e: Event) => {
            const customEvent = e as CustomEvent;
            handleDefinedInteraction(customEvent.detail.interactionType, customEvent.detail.metadata);
        });
        // TOO NOISY: enable if needed but tbh not worth it
        //window.addEventListener('scroll', handleScroll, { passive: true });

        return () => {
            document.removeEventListener('click', handleClick);
            document.removeEventListener('definedInteraction', (e: Event) => {
                const customEvent = e as CustomEvent;
                handleDefinedInteraction(customEvent.detail.interactionType, customEvent.detail.metadata);
            });
            //window.removeEventListener('scroll', handleScroll);
        };
    }, []);
};
