import { useEffect, useRef, useState } from 'react';
import '@/lib/experiments'
import type { EventName } from '@/lib/events';

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
  eventName: EventName;
  page: string;
  productId?: string;
  metadata?: Record<string, unknown>;
}) => {
  try {
    const experimentId = localStorage.getItem('phantom_experiment_id');
    await fetch('/api/ingest', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        ...ev,
        experimentId: experimentId || undefined,
      }),
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

        const handlePageView = () => {
            if (!sidRef.current) return;
            const page = window.location.pathname;
            track({
                sessionId: sidRef.current,
                eventName: 'page_view',
                page,
                metadata: {
                    referrer: document.referrer,
                },
            });
        };

        // called for canonical events dispatched via custom events
        const handleDefinedInteraction = (e: Event) => {
            if (!sidRef.current) return;
            const customEvent = e as CustomEvent<{
                eventName: EventName;
                productId?: string;
                metadata?: Record<string, unknown>;
            }>;
            const page = window.location.pathname;
            track({
                sessionId: sidRef.current,
                eventName: customEvent.detail.eventName,
                page,
                productId: customEvent.detail.productId,
                metadata: customEvent.detail.metadata,
            });
        };

        // wait for session to be ready before tracking
        if (!ready) return;

        handlePageView();
        document.addEventListener('definedInteraction', handleDefinedInteraction);

        return () => {
            document.removeEventListener('definedInteraction', handleDefinedInteraction);
        };
    }, [ready]);
};
