import { z } from 'zod';

// canonical events for tracking user interactions
export type EventName =
  // navigation & discovery
  | 'page_view'
  | 'view_item_page'
  | 'learn_more_about_item'
  // cart operations
  | 'add_item_to_cart'
  | 'remove_item'
  | 'checkout_start'
  | 'purchase_complete'
  // filtering & search
  | 'search'
  | 'filter_for_date'
  | 'filter_for_amenities'
  | 'filter_for_price'
  | 'sort_change'
  // dwell signals (Ns threshold)
  | 'hover_over_title'
  | 'hover_over_paragraph'
  | 'hover_over_link'
  | 'hover_over_button'
  // session
  | 'session_start';

export const eventNames: readonly EventName[] = [
  'page_view',
  'view_item_page',
  'learn_more_about_item',
  'add_item_to_cart',
  'remove_item',
  'checkout_start',
  'purchase_complete',
  'search',
  'filter_for_date',
  'filter_for_amenities',
  'filter_for_price',
  'sort_change',
  'hover_over_title',
  'hover_over_paragraph',
  'hover_over_link',
  'hover_over_button',
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
    'view_item_page',
    'learn_more_about_item',
    'add_item_to_cart',
    'remove_item',
    'checkout_start',
    'purchase_complete',
    'search',
    'filter_for_date',
    'filter_for_amenities',
    'filter_for_price',
    'sort_change',
    'hover_over_title',
    'hover_over_paragraph',
    'hover_over_link',
    'hover_over_button',
    'session_start',
  ]),
  productId: z.string().optional(),
  metadata: z.record(z.string(), z.unknown()).optional(),
  userAgent: z.string().optional(),
});

export type EventBaseValidated = z.infer<typeof eventBaseSchema>;
