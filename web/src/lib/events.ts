import { z } from 'zod';

// canonical events for tracking user interactions
export type EventName =
  | 'page_view'
  | 'click'
  | 'product_view'
  | 'product_hover'
  | 'search'
  | 'filter_apply'
  | 'sort_change'
  | 'add_to_cart'
  | 'remove_from_cart'
  | 'checkout_start'
  | 'purchase_complete'
  | 'session_start';

export const eventNames: readonly EventName[] = [
  'page_view',
  'click',
  'product_view',
  'product_hover',
  'search',
  'filter_apply',
  'sort_change',
  'add_to_cart',
  'remove_from_cart',
  'checkout_start',
  'purchase_complete',
  'session_start',
] as const;

export interface EventBase {
  sessionId: string;
  experimentId?: string;
  storeMode: 'hotel' | 'airline';
  ts: string; // ISO8601
  page: string;
  eventName: EventName;
  productId?: string;
  metadata?: Record<string, unknown>;
  userAgent?: string;
}

// zod schema for runtime validation
export const eventBaseSchema = z.object({
  sessionId: z.string().min(1),
  experimentId: z.string().optional(),
  storeMode: z.enum(['hotel', 'airline']),
  ts: z.string().datetime(), // validates ISO8601
  page: z.string().min(1),
  eventName: z.enum([
    'page_view',
    'click',
    'product_view',
    'product_hover',
    'search',
    'filter_apply',
    'sort_change',
    'add_to_cart',
    'remove_from_cart',
    'checkout_start',
    'purchase_complete',
    'session_start',
  ]),
  productId: z.string().optional(),
  metadata: z.record(z.string(), z.unknown()).optional(),
  userAgent: z.string().optional(),
});

export type EventBaseValidated = z.infer<typeof eventBaseSchema>;
