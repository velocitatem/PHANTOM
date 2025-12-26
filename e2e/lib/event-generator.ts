import { EventPayload, PriceLogPayload } from './api-client';
import { v4 as uuidv4 } from 'uuid';

/**
 * Canonical event names matching the frontend
 */
export const EventNames = {
  // Navigation events
  PAGE_VIEW: 'page_view',
  VIEW_ITEM_PAGE: 'view_item_page',
  LEARN_MORE: 'learn_more_about_item',

  // Cart events
  ADD_TO_CART: 'add_item_to_cart',
  REMOVE_FROM_CART: 'remove_item',
  CHECKOUT_START: 'checkout_start',
  PURCHASE_COMPLETE: 'purchase_complete',

  // Search/Filter events
  SEARCH: 'search',
  FILTER_DATE: 'filter_for_date',
  FILTER_AMENITIES: 'filter_for_amenities',
  FILTER_PRICE: 'filter_for_price',
  SORT_CHANGE: 'sort_change',

  // Dwell signals (engagement)
  HOVER_TITLE: 'hover_over_title',
  HOVER_PARAGRAPH: 'hover_over_paragraph',
  HOVER_LINK: 'hover_over_link',
  HOVER_BUTTON: 'hover_over_button',

  // Session
  SESSION_START: 'session_start',
} as const;

export type EventName = typeof EventNames[keyof typeof EventNames];

/**
 * Test product configuration
 */
export interface TestProduct {
  id: string;
  basePrice: number;
  storeMode: 'hotel' | 'airline';
  name?: string;
}

/**
 * Generates test events for dynamic pricing E2E tests
 */
export class EventGenerator {
  private sessionId: string;
  private experimentId: string;
  private storeMode: 'hotel' | 'airline';

  constructor(options?: {
    sessionId?: string;
    experimentId?: string;
    storeMode?: 'hotel' | 'airline';
  }) {
    this.sessionId = options?.sessionId || uuidv4();
    this.experimentId = options?.experimentId || uuidv4();
    this.storeMode = options?.storeMode || 'hotel';
  }

  get session(): string {
    return this.sessionId;
  }

  get experiment(): string {
    return this.experimentId;
  }

  /**
   * Create a new session for isolation between test scenarios
   */
  newSession(): string {
    this.sessionId = uuidv4();
    return this.sessionId;
  }

  /**
   * Generate a single event
   */
  createEvent(
    eventName: EventName,
    productId: string,
    metadata?: Record<string, unknown>
  ): EventPayload {
    return {
      sessionId: this.sessionId,
      experimentId: this.experimentId,
      eventName,
      page: `/${this.storeMode}/products/${productId}`,
      productId,
      metadata: metadata || {},
      storeMode: this.storeMode,
      userAgent: 'PHANTOM-E2E-Test/1.0',
      ts: new Date().toISOString(),
    };
  }

  /**
   * Generate a product view event
   */
  viewProduct(productId: string): EventPayload {
    return this.createEvent(EventNames.VIEW_ITEM_PAGE, productId, {
      referrer: `/${this.storeMode}/products`,
      viewport: { width: 1920, height: 1080 },
    });
  }

  /**
   * Generate a "learn more" event (high intent signal)
   */
  learnMore(productId: string): EventPayload {
    return this.createEvent(EventNames.LEARN_MORE, productId, {
      section: 'details',
    });
  }

  /**
   * Generate a hover event (engagement signal)
   */
  hover(productId: string, element: 'title' | 'paragraph' | 'button' = 'title'): EventPayload {
    const eventMap = {
      title: EventNames.HOVER_TITLE,
      paragraph: EventNames.HOVER_PARAGRAPH,
      button: EventNames.HOVER_BUTTON,
    };
    return this.createEvent(eventMap[element], productId, {
      duration_ms: Math.floor(Math.random() * 2000) + 500,
    });
  }

  /**
   * Generate an add-to-cart event
   */
  addToCart(productId: string, quantity: number = 1): EventPayload {
    return this.createEvent(EventNames.ADD_TO_CART, productId, {
      quantity,
      cart_size: quantity,
    });
  }

  /**
   * Generate a sequence of high-velocity events for surge pricing trigger
   * This simulates rapid user interest in a product
   */
  generateSurgeSequence(productId: string, count: number): EventPayload[] {
    const events: EventPayload[] = [];

    for (let i = 0; i < count; i++) {
      // Mix of different event types to simulate realistic behavior
      events.push(this.viewProduct(productId));

      if (i % 2 === 0) {
        events.push(this.learnMore(productId));
      }

      if (i % 3 === 0) {
        events.push(this.hover(productId, 'title'));
      }
    }

    return events;
  }

  /**
   * Generate a normal browsing session (not triggering surge)
   */
  generateNormalSession(productId: string): EventPayload[] {
    return [
      this.viewProduct(productId),
      this.hover(productId, 'title'),
    ];
  }

  /**
   * Generate high-velocity agent-like behavior
   * This should trigger SessionAwarePricer's agent detection
   */
  generateAgentBehavior(productIds: string[]): EventPayload[] {
    const events: EventPayload[] = [];

    // Rapid-fire product views across multiple products
    for (const productId of productIds) {
      events.push(this.viewProduct(productId));
      // Very quick succession - agent-like behavior
    }

    return events;
  }

  /**
   * Generate a price log entry
   */
  createPriceLog(productId: string, price: number): PriceLogPayload {
    return {
      productId,
      price,
      sessionId: this.sessionId,
      experimentId: this.experimentId,
      storeMode: this.storeMode,
      ts: new Date().toISOString(),
    };
  }
}

/**
 * Pre-configured test products for E2E tests
 * These should match products in your test database
 */
export const TestProducts = {
  // Hotel products with known base prices
  hotel1: {
    id: 'e2e-test-hotel-001',
    basePrice: 150.00,
    storeMode: 'hotel' as const,
    name: 'E2E Test Hotel 1',
  },
  hotel2: {
    id: 'e2e-test-hotel-002',
    basePrice: 200.00,
    storeMode: 'hotel' as const,
    name: 'E2E Test Hotel 2',
  },
  hotel3: {
    id: 'e2e-test-hotel-003',
    basePrice: 100.00,
    storeMode: 'hotel' as const,
    name: 'E2E Test Hotel 3',
  },

  // Airline products
  airline1: {
    id: 'e2e-test-airline-001',
    basePrice: 350.00,
    storeMode: 'airline' as const,
    name: 'E2E Test Flight 1',
  },
};

/**
 * Generate a unique test product ID for isolation
 */
export function generateTestProductId(prefix: string = 'e2e-test'): string {
  return `${prefix}-${uuidv4().slice(0, 8)}`;
}
