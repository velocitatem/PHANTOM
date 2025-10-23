import { useEffect, useRef } from 'react';

const genSessionId = () => {
  if (typeof window === 'undefined') return '';
  let sid = sessionStorage.getItem('phantom_session_id');
  if (!sid) {
    sid = `${Date.now()}-${Math.random().toString(36).slice(2)}`;
    sessionStorage.setItem('phantom_session_id', sid);
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

    handlePageView();
    document.addEventListener('click', handleClick);
    window.addEventListener('scroll', handleScroll, { passive: true });

    return () => {
      document.removeEventListener('click', handleClick);
      window.removeEventListener('scroll', handleScroll);
    };
  }, []);
};
