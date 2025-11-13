import { useCallback, useRef } from 'react';
import type { EventName } from '@/lib/events';

const dispatchInteraction = (
    eventName: EventName,
    productId?: string,
    metadata?: Record<string, unknown>
) => {
    const e = new CustomEvent('definedInteraction', {
        detail: { eventName, productId, metadata },
    });
    document.dispatchEvent(e);
};

interface UseHoverTrackingOptions {
    eventName: EventName;
    productId?: string;
    metadata?: Record<string, unknown>;
    threshold?: number; // ms, default 1500 or NEXT_PUBLIC_HOVER_THRESHOLD
}

export const useHoverTracking = (options: UseHoverTrackingOptions) => {
    const defaultThreshold = process.env.NEXT_PUBLIC_HOVER_THRESHOLD
        ? parseInt(process.env.NEXT_PUBLIC_HOVER_THRESHOLD, 10)
        : 1500;
    const { eventName, productId, metadata, threshold = defaultThreshold } = options;
    const timerRef = useRef<NodeJS.Timeout | undefined>(undefined);
    const startRef = useRef<number | undefined>(undefined);

    return useCallback((node: HTMLElement | null) => {
        if (!node) {
            if (timerRef.current) clearTimeout(timerRef.current);
            return;
        }

        const onEnter = () => {
            startRef.current = Date.now();
            timerRef.current = setTimeout(() => {
                const dwellTime = Date.now() - startRef.current!;
                dispatchInteraction(eventName, productId, {
                    ...metadata,
                    dwellTime,
                });
            }, threshold);
        };

        const onLeave = () => {
            if (timerRef.current) {
                clearTimeout(timerRef.current);
                timerRef.current = undefined;
            }
        };

        node.addEventListener('mouseenter', onEnter);
        node.addEventListener('mouseleave', onLeave);

        return () => {
            node.removeEventListener('mouseenter', onEnter);
            node.removeEventListener('mouseleave', onLeave);
            if (timerRef.current) clearTimeout(timerRef.current);
        };
    }, [eventName, productId, metadata, threshold]);
};
