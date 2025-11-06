type SessionData = {
  experimentId?: string;
  startedAt: number;
  status: 'active' | 'stopped';
};

type ExperimentData = {
  id: string;
  status: 'active' | 'stopped';
  sessionIds: string[];
  createdAt: number;
};

const store = new Map<string, SessionData>();
const experiments = new Map<string, ExperimentData>();

const cfg = {
  key: process.env.AIRTABLE_API_KEY,
  base: process.env.AIRTABLE_BASE_ID,
  table: process.env.AIRTABLE_TABLE_NAME || 'Sessions',
};

// sync session to airtable if credentials present
const syncToAirtable = async (sid: string, data: SessionData) => {
  if (!cfg.key || !cfg.base) return; // skip if not configured

  try {
    const url = `https://api.airtable.com/v0/${cfg.base}/${encodeURIComponent(cfg.table)}`;
    await fetch(url, {
      method: 'POST',
      headers: {
        Authorization: `Bearer ${cfg.key}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        fields: {
          sessionId: sid,
          experimentId: data.experimentId || '',
          startedAt: new Date(data.startedAt).toISOString(),
          status: data.status,
        },
      }),
    });
  } catch (err) {
    console.error('airtable sync failed:', err);
  }
};

export const getSession = (sid: string) => store.get(sid);

export const createSession = (sid: string) => {
  const data: SessionData = { startedAt: Date.now(), status: 'active' };
  store.set(sid, data);
  syncToAirtable(sid, data); // async fire-and-forget
  return data;
};

export const setExperiment = (sid: string, expId: string) => {
  const data = store.get(sid) || createSession(sid);
  data.experimentId = expId;
  store.set(sid, data);
  syncToAirtable(sid, data);
  return data;
};

export const stopExperiment = (sid: string) => {
  const data = store.get(sid);
  if (data) {
    data.status = 'stopped';
    store.set(sid, data);
    syncToAirtable(sid, data);
  }
  return data;
};

// experiment-level operations
export const createExperiment = (sid: string, expId: string) => {
  const exp: ExperimentData = {
    id: expId,
    status: 'active',
    sessionIds: [sid],
    createdAt: Date.now(),
  };
  experiments.set(expId, exp);
  setExperiment(sid, expId); // link session to experiment
  console.log(`experiment ${expId} started with session ${sid}`);
  return exp;
};

export const stopExperimentById = (expId: string) => {
  const exp = experiments.get(expId);
  if (exp) {
    exp.status = 'stopped';
    experiments.set(expId, exp);
    console.log(`experiment ${expId} stopped`);
  }
  return exp;
};

export const getExperiment = (expId: string) => experiments.get(expId);

export const getAllExperiments = () => Array.from(experiments.values());
