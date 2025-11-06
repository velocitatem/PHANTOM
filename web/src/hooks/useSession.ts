import { useEffect, useState } from 'react';

type SessionState = {
  sessionId: string | null;
  experimentId: string | null;
  isLoading: boolean;
};

export const useSession = () => {
  const [state, setState] = useState<SessionState>({
    sessionId: null,
    experimentId: null,
    isLoading: true,
  });

  useEffect(() => {
    const fetchSession = async () => {
      try {
        const res = await fetch('/api/session');
        if (!res.ok) throw new Error(`fetch failed: ${res.status}`);

        const data = await res.json();
        setState({
          sessionId: data.sessionId || null,
          experimentId: data.experimentId || null,
          isLoading: false,
        });
      } catch (err) {
        console.error('session fetch error:', err);
        setState({ sessionId: null, experimentId: null, isLoading: false });
      }
    };

    fetchSession();
  }, []);

  return state;
};
