interface InteractionEvent {
  sessionId: string;
  event: string;
  productId?: string;
  timestamp: string;
  metadata?: Record<string, any>;
}

const dumpKafkaTopic = async (backendUrl: string, topic: string) => {
  const resp = await fetch(`${backendUrl}/api/kafka/dump?topic=${topic}`);
  if (!resp.ok) throw new Error(`Kafka dump failed: ${resp.status}`);
  const { data = [] } = await resp.json();
  return data as any[];
};

export const waitForInteractionEvent = async (
  backendUrl: string,
  sessionId: string,
  eventType: string,
  maxRetries = 10,
  pollInterval = 500
): Promise<InteractionEvent | null> => {
  for (let i = 0; i < maxRetries; i++) {
    const msgs = await dumpKafkaTopic(backendUrl, "user-interactions");
    const hit = msgs.find(m => m.sessionId === sessionId && m.event === eventType);
    if (hit) return hit as InteractionEvent;
    await new Promise<void>(r => setTimeout(r, pollInterval));
  }
  return null;
};

export const countProductViews = async (backendUrl: string, productId: string) =>
  (await dumpKafkaTopic(backendUrl, "user-interactions")).reduce(
    (n, m) => n + (m.productId === productId && m.event === "view_item_page" ? 1 : 0),
    0
  );

export const getSessionEvents = async (backendUrl: string, sessionId: string) =>
  (await dumpKafkaTopic(backendUrl, "user-interactions")).filter(m => m.sessionId === sessionId);
