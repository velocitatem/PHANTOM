'use client';

import { useEffect, useState } from 'react';
import { useSession } from '@/hooks/useSession';

type Experiment = {
  id: string;
  status: 'active' | 'stopped';
  sessionIds: string[];
  createdAt: number;
};

export default function ExperimentsAdmin() {
  const { sessionId, isLoading: sessionLoading } = useSession();
  const [exps, setExps] = useState<Experiment[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchExps = async () => {
    try {
      const res = await fetch('/api/admin/experiments');
      if (!res.ok) throw new Error(`fetch failed: ${res.status}`);
      const data = await res.json();
      setExps(data.experiments || []);
    } catch (err: any) {
      setError(err.message);
    }
  };

  useEffect(() => {
    fetchExps();
  }, []);

  const handleStart = async () => {
    if (!sessionId) {
      setError('no session available');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const res = await fetch('/api/admin/experiments/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ sessionId }),
      });

      if (!res.ok) {
        const data = await res.json();
        throw new Error(data.error || 'start failed');
      }

      await fetchExps(); // refresh list
    } catch (err: any) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleStop = async (expId: string) => {
    setLoading(true);
    setError(null);

    try {
      const res = await fetch('/api/admin/experiments/stop', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ experimentId: expId }),
      });

      if (!res.ok) {
        const data = await res.json();
        throw new Error(data.error || 'stop failed');
      }

      await fetchExps(); // refresh list
    } catch (err: any) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  if (sessionLoading) {
    return (
      <div className="flex min-h-screen items-center justify-center bg-zinc-50 dark:bg-black">
        <p className="text-zinc-600 dark:text-zinc-400">loading session...</p>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-zinc-50 px-6 py-12 dark:bg-black">
      <div className="mx-auto max-w-5xl">
        <div className="mb-8 flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-semibold tracking-tight text-black dark:text-zinc-50">
              Experiments
            </h1>
            <p className="mt-2 text-sm text-zinc-600 dark:text-zinc-400">
              current session: {sessionId || 'none'}
            </p>
          </div>
          <button
            onClick={handleStart}
            disabled={loading || !sessionId}
            className="rounded-lg bg-black px-4 py-2 text-sm font-medium text-white transition-colors hover:bg-zinc-800 disabled:opacity-50 dark:bg-zinc-50 dark:text-black dark:hover:bg-zinc-200"
          >
            {loading ? 'starting...' : 'start experiment'}
          </button>
        </div>

        {error && (
          <div className="mb-4 rounded-lg bg-red-50 p-4 text-sm text-red-800 dark:bg-red-950 dark:text-red-200">
            {error}
          </div>
        )}

        <div className="overflow-hidden rounded-lg border border-zinc-200 bg-white dark:border-zinc-800 dark:bg-zinc-950">
          <table className="w-full text-left text-sm">
            <thead className="border-b border-zinc-200 bg-zinc-50 dark:border-zinc-800 dark:bg-zinc-900">
              <tr>
                <th className="px-6 py-3 font-medium text-zinc-900 dark:text-zinc-100">
                  experiment id
                </th>
                <th className="px-6 py-3 font-medium text-zinc-900 dark:text-zinc-100">
                  status
                </th>
                <th className="px-6 py-3 font-medium text-zinc-900 dark:text-zinc-100">
                  session count
                </th>
                <th className="px-6 py-3 font-medium text-zinc-900 dark:text-zinc-100">
                  created
                </th>
                <th className="px-6 py-3 font-medium text-zinc-900 dark:text-zinc-100">
                  action
                </th>
              </tr>
            </thead>
            <tbody className="divide-y divide-zinc-200 dark:divide-zinc-800">
              {exps.length === 0 ? (
                <tr>
                  <td
                    colSpan={5}
                    className="px-6 py-8 text-center text-zinc-500 dark:text-zinc-400"
                  >
                    no experiments yet
                  </td>
                </tr>
              ) : (
                exps.map((exp) => (
                  <tr
                    key={exp.id}
                    className="hover:bg-zinc-50 dark:hover:bg-zinc-900"
                  >
                    <td className="px-6 py-4 font-mono text-xs text-zinc-700 dark:text-zinc-300">
                      {exp.id.slice(0, 8)}...
                    </td>
                    <td className="px-6 py-4">
                      <span
                        className={`inline-block rounded-full px-2 py-1 text-xs font-medium ${
                          exp.status === 'active'
                            ? 'bg-green-100 text-green-800 dark:bg-green-950 dark:text-green-200'
                            : 'bg-zinc-100 text-zinc-800 dark:bg-zinc-800 dark:text-zinc-200'
                        }`}
                      >
                        {exp.status}
                      </span>
                    </td>
                    <td className="px-6 py-4 text-zinc-700 dark:text-zinc-300">
                      {exp.sessionIds.length}
                    </td>
                    <td className="px-6 py-4 text-zinc-700 dark:text-zinc-300">
                      {new Date(exp.createdAt).toLocaleString()}
                    </td>
                    <td className="px-6 py-4">
                      {exp.status === 'active' && (
                        <button
                          onClick={() => handleStop(exp.id)}
                          disabled={loading}
                          className="text-sm font-medium text-red-600 hover:text-red-700 disabled:opacity-50 dark:text-red-400 dark:hover:text-red-300"
                        >
                          stop
                        </button>
                      )}
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
