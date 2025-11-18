'use client';

import { useEffect, useState, Suspense } from 'react';
import { useSearchParams, useRouter } from 'next/navigation';

const StartTaskContent = () => {
  const searchParams = useSearchParams();
  const router = useRouter();
  const [status, setStatus] = useState<'loading' | 'error' | 'redirecting'>('loading');
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const uuid = searchParams.get('uuid');

    if (!uuid) {
      setError('no experiment UUID provided');
      setStatus('error');
      return;
    }

    const validateAndStore = async () => {
      try {
        const res = await fetch(`/api/admin/experiments?id=${uuid}`);
        if (!res.ok) throw new Error('experiment not found');

        const data = await res.json();
        const exp = data.experiment;

        if (!exp) throw new Error('invalid experiment UUID');

        localStorage.setItem('phantom_experiment_id', uuid);

        await fetch('/api/session', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ experimentId: uuid }),
        });

        setStatus('redirecting');

        setTimeout(() => {
          router.push("/");
        }, 800);

      } catch (err: any) {
        setError(err.message || 'failed to start task');
        setStatus('error');
      }
    };

    validateAndStore();
  }, [searchParams, router]);

  return (
    <div className="flex min-h-screen items-center justify-center bg-zinc-50 dark:bg-black">
      <div className="text-center">
        {status === 'loading' && (
          <div>
            <div className="mb-4 h-8 w-8 animate-spin rounded-full border-4 border-zinc-200 border-t-zinc-900 dark:border-zinc-800 dark:border-t-zinc-100 mx-auto" />
            <p className="text-zinc-600 dark:text-zinc-400">validating browser...</p>
          </div>
        )}

        {status === 'redirecting' && (
          <div>
            <div className="mb-4 text-4xl">✓</div>
            <p className="text-zinc-900 dark:text-zinc-100 font-medium">website loaded</p>
            <p className="mt-2 text-sm text-zinc-600 dark:text-zinc-400">redirecting to page...</p>
          </div>
        )}

        {status === 'error' && (
          <div className="rounded-lg bg-red-50 p-6 dark:bg-red-950">
            <p className="text-red-900 dark:text-red-100 font-medium">error</p>
            <p className="mt-2 text-sm text-red-700 dark:text-red-300">{error}</p>
          </div>
        )}
      </div>
    </div>
  );
};

export default function StartTaskPage() {
  return (
    <Suspense fallback={
      <div className="flex min-h-screen items-center justify-center bg-zinc-50 dark:bg-black">
        <p className="text-zinc-600 dark:text-zinc-400">loading...</p>
      </div>
    }>
      <StartTaskContent />
    </Suspense>
  );
}
