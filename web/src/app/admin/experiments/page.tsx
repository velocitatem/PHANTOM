'use client';

import { useEffect, useState } from 'react';
import { TaskManager } from '@/components/admin/TaskManager';
import { ExperimentForm } from '@/components/admin/ExperimentForm';

type Experiment = {
  id: string;
  subject_name: string;
  xp_human_only: boolean;
  xp_market_mode: string;
  created_at: string;
  task?: {
    id: string;
    task_name: string;
  };
};

export default function ExperimentsAdmin() {
  const [exps, setExps] = useState<Experiment[]>([]);
  const [selectedTaskId, setSelectedTaskId] = useState<string | undefined>();
  const [error, setError] = useState<string | null>(null);
  const [showForm, setShowForm] = useState(false);

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

  const handleExperimentCreated = async () => {
    setShowForm(false);
    setSelectedTaskId(undefined);
    await fetchExps();
  };

  return (
    <div className="min-h-screen bg-zinc-50 px-6 py-12 dark:bg-black">
      <div className="mx-auto max-w-7xl">
        <div className="mb-8">
          <h1 className="text-3xl font-semibold tracking-tight text-black dark:text-zinc-50">
            Experiment Management
          </h1>
          <p className="mt-2 text-sm text-zinc-600 dark:text-zinc-400">
            configure tasks and run experiments
          </p>
        </div>

        {error && (
          <div className="mb-4 rounded-lg bg-red-50 p-4 text-sm text-red-800 dark:bg-red-950 dark:text-red-200">
            {error}
          </div>
        )}

        <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
          {/* left column: task manager */}
          <div className="lg:col-span-1">
            <TaskManager
              onTaskSelect={setSelectedTaskId}
              selectedTaskId={selectedTaskId}
            />
          </div>

          {/* right column: experiment form + list */}
          <div className="space-y-6 lg:col-span-2">
            <div className="flex items-center justify-between">
              <h2 className="text-lg font-semibold text-zinc-900 dark:text-zinc-100">
                Experiments
              </h2>
              <button
                onClick={() => setShowForm(!showForm)}
                className="rounded-lg bg-black px-4 py-2 text-sm font-medium text-white transition-colors hover:bg-zinc-800 dark:bg-zinc-50 dark:text-black dark:hover:bg-zinc-200"
              >
                {showForm ? 'hide form' : 'new experiment'}
              </button>
            </div>

            {showForm && (
              <ExperimentForm
                selectedTaskId={selectedTaskId}
                onSuccess={handleExperimentCreated}
              />
            )}

            <div className="overflow-hidden rounded-lg border border-zinc-200 bg-white dark:border-zinc-800 dark:bg-zinc-950">
              <table className="w-full text-left text-sm">
                <thead className="border-b border-zinc-200 bg-zinc-50 dark:border-zinc-800 dark:bg-zinc-900">
                  <tr>
                    <th className="px-4 py-3 font-medium text-zinc-900 dark:text-zinc-100">
                      subject
                    </th>
                    <th className="px-4 py-3 font-medium text-zinc-900 dark:text-zinc-100">
                      mode
                    </th>
                    <th className="px-4 py-3 font-medium text-zinc-900 dark:text-zinc-100">
                      human
                    </th>
                    <th className="px-4 py-3 font-medium text-zinc-900 dark:text-zinc-100">
                      task
                    </th>
                    <th className="px-4 py-3 font-medium text-zinc-900 dark:text-zinc-100">
                      created
                    </th>
                    <th className="px-4 py-3 font-medium text-zinc-900 dark:text-zinc-100">
                      link
                    </th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-zinc-200 dark:divide-zinc-800">
                  {exps.length === 0 ? (
                    <tr>
                      <td
                        colSpan={6}
                        className="px-4 py-8 text-center text-zinc-500 dark:text-zinc-400"
                      >
                        no experiments yet
                      </td>
                    </tr>
                  ) : (
                    exps.map((exp) => {
                      const baseUrl = exp.xp_market_mode === 'airline'
                        ? 'https://phantom-airline.vercel.app'
                        : 'https://phantom-hotel.vercel.app';
                      const link = `${baseUrl}/start-task?uuid=${exp.id}`;

                      return (
                        <tr
                          key={exp.id}
                          className="hover:bg-zinc-50 dark:hover:bg-zinc-900"
                        >
                          <td className="px-4 py-3 font-medium text-zinc-900 dark:text-zinc-100">
                            {exp.subject_name}
                          </td>
                          <td className="px-4 py-3">
                            <span className="inline-block rounded-full bg-zinc-100 px-2 py-1 text-xs font-medium text-zinc-800 dark:bg-zinc-800 dark:text-zinc-200">
                              {exp.xp_market_mode || 'none'}
                            </span>
                          </td>
                          <td className="px-4 py-3">
                            {exp.xp_human_only ? (
                              <span className="text-xs text-green-600 dark:text-green-400">
                                yes
                              </span>
                            ) : (
                              <span className="text-xs text-zinc-500">no</span>
                            )}
                          </td>
                          <td className="px-4 py-3 text-xs text-zinc-600 dark:text-zinc-400">
                            {exp.task ? exp.task.task_name : '—'}
                          </td>
                          <td className="px-4 py-3 text-xs text-zinc-600 dark:text-zinc-400">
                            {new Date(exp.created_at).toLocaleDateString()}
                          </td>
                          <td className="px-4 py-3">
                            <button
                              onClick={() => {
                                navigator.clipboard.writeText(link);
                              }}
                              className="text-xs font-medium text-zinc-900 hover:text-zinc-600 dark:text-zinc-100 dark:hover:text-zinc-400"
                            >
                              copy link
                            </button>
                          </td>
                        </tr>
                      );
                    })
                  )}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
