import { testConfig } from '../playwright.config';

/**
 * Event payload structure matching the backend API
 */
export interface EventPayload {
  sessionId: string;
  experimentId?: string;
  eventName: string;
  page: string;
  productId?: string;
  metadata?: Record<string, unknown>;
  storeMode: 'hotel' | 'airline';
  userAgent?: string;
  ts?: string;
}

/**
 * Price log payload structure
 */
export interface PriceLogPayload {
  productId: string;
  price: number;
  sessionId: string;
  experimentId?: string;
  storeMode: 'hotel' | 'airline';
  ts?: string;
}

/**
 * Price response from the pricing provider
 */
export interface PriceResponse {
  productId: string;
  price: number;
  base_price: number;
  markup: number;
  elasticity: number | null;
  model_version: string;
}

/**
 * API client for interacting with PHANTOM services
 */
export class PhantomApiClient {
  private backendUrl: string;
  private providerUrl: string;

  constructor(
    backendUrl: string = testConfig.backendUrl,
    providerUrl: string = testConfig.providerUrl
  ) {
    this.backendUrl = backendUrl;
    this.providerUrl = providerUrl;
  }

  /**
   * Send a user interaction event to the ingestion API
   */
  async ingestEvent(event: EventPayload): Promise<{ success: boolean }> {
    const payload: EventPayload = {
      ...event,
      ts: event.ts || new Date().toISOString(),
    };

    const response = await fetch(`${this.backendUrl}/api/kafka/ingest`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });

    if (!response.ok) {
      throw new Error(`Failed to ingest event: ${response.status} ${await response.text()}`);
    }

    return response.json();
  }

  /**
   * Send multiple events in rapid succession
   */
  async ingestEvents(events: EventPayload[], delayMs: number = 100): Promise<void> {
    for (const event of events) {
      await this.ingestEvent(event);
      if (delayMs > 0) {
        await new Promise(resolve => setTimeout(resolve, delayMs));
      }
    }
  }

  /**
   * Log a price observation
   */
  async logPrice(priceLog: PriceLogPayload): Promise<{ success: boolean }> {
    const payload: PriceLogPayload = {
      ...priceLog,
      ts: priceLog.ts || new Date().toISOString(),
    };

    const response = await fetch(`${this.backendUrl}/api/kafka/price-log`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });

    if (!response.ok) {
      throw new Error(`Failed to log price: ${response.status} ${await response.text()}`);
    }

    return response.json();
  }

  /**
   * Get the current price for a product from the pricing provider
   */
  async getPrice(
    mode: 'hotel' | 'airline',
    productId: string,
    sessionId?: string
  ): Promise<PriceResponse> {
    const params = new URLSearchParams();
    if (sessionId) {
      params.set('sessionId', sessionId);
    }

    const url = `${this.providerUrl}/api/${mode}/price/${productId}${params.toString() ? '?' + params.toString() : ''}`;
    const response = await fetch(url);

    if (!response.ok) {
      throw new Error(`Failed to get price: ${response.status} ${await response.text()}`);
    }

    return response.json();
  }

  /**
   * Dump events from Kafka topic for debugging
   */
  async dumpKafkaEvents(
    topic: 'user-interactions' | 'price-logs' = 'user-interactions',
    lastN?: number
  ): Promise<{ success: boolean; count: number; data: unknown[] }> {
    const params = new URLSearchParams({ topic });
    if (lastN) {
      params.set('last_n', String(lastN));
    }

    const response = await fetch(`${this.backendUrl}/api/kafka/dump?${params.toString()}`);

    if (!response.ok) {
      throw new Error(`Failed to dump Kafka events: ${response.status}`);
    }

    return response.json();
  }

  /**
   * Check health of backend service
   */
  async checkBackendHealth(): Promise<{ status: string; kafka: string }> {
    const response = await fetch(`${this.backendUrl}/health`);
    return response.json();
  }

  /**
   * Check health of pricing provider
   */
  async checkProviderHealth(): Promise<{ status: string; redis: boolean }> {
    const response = await fetch(`${this.providerUrl}/health`);
    return response.json();
  }

  /**
   * List registered models in the pricing provider
   */
  async listModels(): Promise<Record<string, unknown>> {
    const response = await fetch(`${this.providerUrl}/models`);
    return response.json();
  }

  /**
   * Reload models in the pricing provider
   */
  async reloadModels(): Promise<{ elasticity_loaded: boolean; pricing_model_loaded: boolean }> {
    const response = await fetch(`${this.providerUrl}/models/reload`, { method: 'POST' });
    return response.json();
  }
}

// Singleton instance for convenience
export const apiClient = new PhantomApiClient();
