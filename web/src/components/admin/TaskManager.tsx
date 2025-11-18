'use client';

import { useState, useEffect } from 'react';

type Task = {
  id: string;
  task_name: string;
  task_description: string;
  task_def_of_done: string;
  created_at: string;
};

type TaskManagerProps = {
  onTaskSelect?: (taskId: string) => void;
  selectedTaskId?: string;
};

export const TaskManager = ({ onTaskSelect, selectedTaskId }: TaskManagerProps) => {
  const [tasks, setTasks] = useState<Task[]>([]);
  const [loading, setLoading] = useState(false);
  const [showForm, setShowForm] = useState(false);
  const [form, setForm] = useState({
    task_name: '',
    task_description: '',
    task_def_of_done: '',
  });
  const [error, setError] = useState<string | null>(null);

  const fetchTasks = async () => {
    try {
      const res = await fetch('/api/admin/tasks');
      if (!res.ok) throw new Error(`fetch failed: ${res.status}`);
      const data = await res.json();
      setTasks(data.tasks || []);
    } catch (err: any) {
      setError(err.message);
    }
  };

  useEffect(() => {
    fetchTasks();
  }, []);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError(null);

    try {
      const res = await fetch('/api/admin/tasks', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(form),
      });

      if (!res.ok) {
        const data = await res.json();
        throw new Error(data.error || 'creation failed');
      }

      setForm({ task_name: '', task_description: '', task_def_of_done: '' });
      setShowForm(false);
      await fetchTasks();
    } catch (err: any) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-semibold text-zinc-900 dark:text-zinc-100">
          Tasks
        </h2>
        <button
          onClick={() => setShowForm(!showForm)}
          className="rounded-lg bg-zinc-900 px-3 py-1.5 text-sm font-medium text-white transition-colors hover:bg-zinc-700 dark:bg-zinc-100 dark:text-black dark:hover:bg-zinc-300"
        >
          {showForm ? 'cancel' : 'new task'}
        </button>
      </div>

      {error && (
        <div className="rounded-lg bg-red-50 p-3 text-sm text-red-800 dark:bg-red-950 dark:text-red-200">
          {error}
        </div>
      )}

      {showForm && (
        <form onSubmit={handleSubmit} className="space-y-3 rounded-lg border border-zinc-200 bg-white p-4 dark:border-zinc-800 dark:bg-zinc-950">
          <div>
            <label className="block text-sm font-medium text-zinc-700 dark:text-zinc-300">
              task name
            </label>
            <input
              type="text"
              value={form.task_name}
              onChange={(e) => setForm({ ...form, task_name: e.target.value })}
              className="mt-1 w-full rounded-lg border border-zinc-300 bg-white px-3 py-2 text-sm text-zinc-900 focus:border-zinc-900 focus:outline-none dark:border-zinc-700 dark:bg-zinc-900 dark:text-zinc-100 dark:focus:border-zinc-100"
              placeholder="e.g., Book cheapest flight to Paris"
              required
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-zinc-700 dark:text-zinc-300">
              description
            </label>
            <textarea
              value={form.task_description}
              onChange={(e) => setForm({ ...form, task_description: e.target.value })}
              className="mt-1 w-full rounded-lg border border-zinc-300 bg-white px-3 py-2 text-sm text-zinc-900 focus:border-zinc-900 focus:outline-none dark:border-zinc-700 dark:bg-zinc-900 dark:text-zinc-100 dark:focus:border-zinc-100"
              placeholder="User should find and book the cheapest available flight..."
              rows={3}
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-zinc-700 dark:text-zinc-300">
              definition of done
            </label>
            <textarea
              value={form.task_def_of_done}
              onChange={(e) => setForm({ ...form, task_def_of_done: e.target.value })}
              className="mt-1 w-full rounded-lg border border-zinc-300 bg-white px-3 py-2 text-sm text-zinc-900 focus:border-zinc-900 focus:outline-none dark:border-zinc-700 dark:bg-zinc-900 dark:text-zinc-100 dark:focus:border-zinc-100"
              placeholder="Booking is completed and confirmation page is shown"
              rows={2}
            />
          </div>

          <button
            type="submit"
            disabled={loading}
            className="w-full rounded-lg bg-black px-4 py-2 text-sm font-medium text-white transition-colors hover:bg-zinc-800 disabled:opacity-50 dark:bg-zinc-50 dark:text-black dark:hover:bg-zinc-200"
          >
            {loading ? 'creating...' : 'create task'}
          </button>
        </form>
      )}

      <div className="space-y-2">
        {tasks.length === 0 ? (
          <p className="py-8 text-center text-sm text-zinc-500 dark:text-zinc-400">
            no tasks yet
          </p>
        ) : (
          tasks.map((task) => (
            <div
              key={task.id}
              onClick={() => onTaskSelect?.(task.id)}
              className={`cursor-pointer rounded-lg border p-3 transition-colors ${
                selectedTaskId === task.id
                  ? 'border-zinc-900 bg-zinc-50 dark:border-zinc-100 dark:bg-zinc-900'
                  : 'border-zinc-200 bg-white hover:border-zinc-300 dark:border-zinc-800 dark:bg-zinc-950 dark:hover:border-zinc-700'
              }`}
            >
              <h3 className="font-medium text-zinc-900 dark:text-zinc-100">
                {task.task_name}
              </h3>
              {task.task_description && (
                <p className="mt-1 text-sm text-zinc-600 dark:text-zinc-400">
                  {task.task_description}
                </p>
              )}
              {task.task_def_of_done && (
                <p className="mt-1 text-xs text-zinc-500 dark:text-zinc-500">
                  done: {task.task_def_of_done}
                </p>
              )}
            </div>
          ))
        )}
      </div>
    </div>
  );
};
