interface InteractionEvent {
  sessionId: string;
  event: string;
  productId?: string;
  timestamp: string;
  metadata?: Record<string, any>;
}

export async function dumpKafkaTopic(
  backendUrl: string,
  topic: string
): Promise<any[]> {
  const resp = await fetch(`${backendUrl}/api/kafka/dump?topic=${topic}`);
  if (!resp.ok) {
    throw new Error(`Kafka dump failed: ${resp.status}`);
  }
  const data = await resp.json();
  return data.messages || [];
}

export async function waitForInteractionEvent(
  backendUrl: string,
  sessionId: string,
  eventType: string,
  maxRetries: number = 10,
  pollInterval: number = 500
): Promise<InteractionEvent | null> {
  for (let i = 0; i < maxRetries; i++) {
    const msgs = await dumpKafkaTopic(backendUrl, 'user-interactions');

    for (const msg of msgs) {
      if (msg.sessionId === sessionId && msg.event === eventType) {
        return msg as InteractionEvent;
      }
    }

    await new Promise(r => setTimeout(r, pollInterval));
  }

  return null;
}

export async function countProductViews(
  backendUrl: string,
  productId: string
): Promise<number> {
  const msgs = await dumpKafkaTopic(backendUrl, 'user-interactions');

  return msgs.reduce((cnt, msg) => {
    if (msg.productId === productId && msg.event === 'view_item_page') {
      return cnt + 1;
    }
    return cnt;
  }, 0);
}

export async function getSessionEvents(
  backendUrl: string,
  sessionId: string
): Promise<InteractionEvent[]> {
  const msgs = await dumpKafkaTopic(backendUrl, 'user-interactions');
  return msgs.filter(m => m.sessionId === sessionId);
}
