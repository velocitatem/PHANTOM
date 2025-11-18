'use client';

import { useState } from 'react';

type ExperimentFormProps = {
  selectedTaskId?: string;
  onSuccess?: () => void;
};

export const ExperimentForm = ({ selectedTaskId, onSuccess }: ExperimentFormProps) => {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [form, setForm] = useState({
    subject_name: '',
    xp_human_only: false,
    xp_market_mode: 'hotel' as 'hotel' | 'airline',
  });

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError(null);

    try {
      const res = await fetch('/api/admin/experiments', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          ...form,
          xp_task_id: selectedTaskId || null,
        }),
      });

      if (!res.ok) {
        const data = await res.json();
        throw new Error(data.error || 'creation failed');
      }

      setForm({ subject_name: '', xp_human_only: false, xp_market_mode: 'hotel' });
      onSuccess?.();
    } catch (err: any) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <form onSubmit={handleSubmit} className="space-y-4 rounded-lg border border-zinc-200 bg-white p-6 dark:border-zinc-800 dark:bg-zinc-950">
      <h2 className="text-lg font-semibold text-zinc-900 dark:text-zinc-100">
        Create Experiment
      </h2>

      {error && (
        <div className="rounded-lg bg-red-50 p-3 text-sm text-red-800 dark:bg-red-950 dark:text-red-200">
          {error}
        </div>
      )}

      <div>
        <label className="block text-sm font-medium text-zinc-700 dark:text-zinc-300">
          subject name
        </label>
        <input
          type="text"
          value={form.subject_name}
          onChange={(e) => setForm({ ...form, subject_name: e.target.value })}
          className="mt-1 w-full rounded-lg border border-zinc-300 bg-white px-3 py-2 text-sm text-zinc-900 focus:border-zinc-900 focus:outline-none dark:border-zinc-700 dark:bg-zinc-900 dark:text-zinc-100 dark:focus:border-zinc-100"
          placeholder="e.g., baseline_dynamic_pricing_v1"
          required
        />
      </div>

      <div>
        <label className="block text-sm font-medium text-zinc-700 dark:text-zinc-300">
          market mode
        </label>
        <select
          value={form.xp_market_mode}
          onChange={(e) => setForm({ ...form, xp_market_mode: e.target.value as 'hotel' | 'airline' })}
          className="mt-1 w-full rounded-lg border border-zinc-300 bg-white px-3 py-2 text-sm text-zinc-900 focus:border-zinc-900 focus:outline-none dark:border-zinc-700 dark:bg-zinc-900 dark:text-zinc-100 dark:focus:border-zinc-100"
        >
          <option value="hotel">hotel</option>
          <option value="airline">airline</option>
        </select>
      </div>

      <div className="flex items-center gap-2">
        <input
          type="checkbox"
          id="human-only"
          checked={form.xp_human_only}
          onChange={(e) => setForm({ ...form, xp_human_only: e.target.checked })}
          className="h-4 w-4 rounded border-zinc-300 text-zinc-900 focus:ring-zinc-900 dark:border-zinc-700 dark:bg-zinc-900"
        />
        <label htmlFor="human-only" className="text-sm text-zinc-700 dark:text-zinc-300">
          human participants only
        </label>
      </div>

      {selectedTaskId && (
        <div className="rounded-lg bg-zinc-50 p-3 dark:bg-zinc-900">
          <p className="text-sm text-zinc-600 dark:text-zinc-400">
            task selected: <span className="font-mono text-xs">{selectedTaskId.slice(0, 8)}...</span>
          </p>
        </div>
      )}

      <button
        type="submit"
        disabled={loading}
        className="w-full rounded-lg bg-black px-4 py-2 text-sm font-medium text-white transition-colors hover:bg-zinc-800 disabled:opacity-50 dark:bg-zinc-50 dark:text-black dark:hover:bg-zinc-200"
      >
        {loading ? 'creating experiment...' : 'create experiment'}
      </button>
    </form>
  );
};
