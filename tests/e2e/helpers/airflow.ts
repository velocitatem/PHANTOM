const AIRFLOW_URL = process.env.AIRFLOW_URL || 'http://localhost:8085';
const AUTH = 'Basic ' + Buffer.from(`${process.env.AIRFLOW_USER || 'admin'}:${process.env.AIRFLOW_PASS || 'admin'}`).toString('base64');

const req = (path: string, opts: any = {}) => {
  const headers = { Authorization: AUTH, ...opts.headers };
  return fetch(`${AIRFLOW_URL}${path}`, { ...opts, headers });
};

export const triggerDag = async (dagId: string, conf = {}) => {
  const r = await req(`/api/v1/dags/${dagId}/dagRuns`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ conf }),
  });
  if (!r.ok) throw new Error(`Trigger DAG failed: ${r.status}`);
  return (await r.json()).dag_run_id;
};

export const getDagStatus = async (dagId: string, runId: string) => {
  const r = await req(`/api/v1/dags/${dagId}/dagRuns/${runId}`);
  if (!r.ok) throw new Error(`Get status failed: ${r.status}`);
  return (await r.json()).state;
};

export const cancelDag = async (dagId: string, runId: string) => {
  const r = await req(`/api/v1/dags/${dagId}/dagRuns/${runId}`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ state: 'failed' }),
  });
  if (!r.ok) console.warn(`Failed to cancel DAG ${runId}: ${r.status}`);
};

export const waitForDag = async (dagId: string, runId: string, maxMs = 30000, pollMs = 1000) => {
  const t0 = Date.now();
  while (Date.now() - t0 < maxMs) {
    const state = await getDagStatus(dagId, runId);
    if (state === 'success') return;
    if (state === 'failed') throw new Error(`DAG ${runId} failed`);
    await new Promise(r => setTimeout(r, pollMs));
  }
  await cancelDag(dagId, runId);
  throw new Error(`DAG ${runId} timeout`);
};

export const runDag = async (dagId: string, conf = {}, maxMs = 60000) => {
  const runId = await triggerDag(dagId, conf);
  await waitForDag(dagId, runId, maxMs);
};

export const runSessionPricing = (mode = 'hotel') =>
  runDag('session_pricing_pipeline', { store_mode: mode, session_limit: 10 }, 90000);

export const runSurgePricing = (mode = 'hotel', highThresh = 10, lowThresh = 2) =>
  runDag('surge_pricing_pipeline', {
    store_mode: mode,
    high_threshold: highThresh,
    low_threshold: lowThresh,
    surge_multiplier: 1.2,
    discount_multiplier: 0.9
  }, 90000);
