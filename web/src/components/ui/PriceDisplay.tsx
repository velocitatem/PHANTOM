'use client';

import { useEffect, useState, useRef } from 'react';

interface PriceDisplayProps {
    productId: string;
    className?: string;
    perNight?: boolean;
}

interface PricingData {
    price: number;
    currency: string;
    cachedAt: string;
}

interface SessionData {
    sessionId: string;
    experimentId?: string;
}

const fetchSession = async (): Promise<SessionData> => {
    try {
        const res = await fetch('/api/session');
        const data = await res.json();
        return {
            sessionId: data.sessionId || '',
            experimentId: data.experimentId || '',
        };
    } catch (err) {
        console.error('failed to fetch session:', err);
        return { sessionId: '', experimentId: '' };
    }
};

const formatPrice = (price: number, currency: string) => {
    return new Intl.NumberFormat('en-US', { // like an std localization
        style: 'currency',
        currency,
    }).format(price);
};

const isCacheStale = (cachedAt: string, thresholdMs = 60000) => {
    const cacheTime = new Date(cachedAt).getTime();
    const now = Date.now();
    return now - cacheTime > thresholdMs;
};

export default function PriceDisplay({
    productId,
    className = '',
    perNight = false,
}: PriceDisplayProps) {
    const sessionRef = useRef<SessionData | null>(null);
    const [data, setData] = useState<PricingData | null>(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);

    useEffect(() => {
        const initAndFetch = async () => {
            setLoading(true);
            setError(null);

            try {
                // fetch session if not already loaded
                if (!sessionRef.current) {
                    sessionRef.current = await fetchSession();
                }

                const { sessionId, experimentId } = sessionRef.current;

                if (!sessionId) {
                    setError('Invalid session');
                    setLoading(false);
                    return;
                }

                const params = new URLSearchParams({
                    productId,
                    sessionId,
                    experimentId: experimentId || '',
                });

                const res = await fetch(`/api/pricing?${params.toString()}`);

                if (!res.ok) {
                    throw new Error(`Failed to fetch price: ${res.status}`);
                }

                const pricingData: PricingData = await res.json();
                setData(pricingData);
            } catch (err) {
                setError(err instanceof Error ? err.message : 'Unknown error');
            } finally {
                setLoading(false);
            }
        };

        initAndFetch();
    }, [productId]);

    if (loading) {
        return (
            <div className={`price-loading ${className}`}>
                <div className="spinner-border animate-spin inline-block w-4 h-4 border-2 rounded-full" role="status">
                    <span className="sr-only">Loading...</span>
                </div>
            </div>
        );
    }

    if (error || !data) {
        return (
            <div className={`price-error ${className}`}>
                <span className="text-red-500 text-sm">Price unavailable</span>
            </div>
        );
    }

    const isStale = isCacheStale(data.cachedAt);
    const formattedPrice = formatPrice(data.price, data.currency);

    return (
        <div className={`price-display ${className}`}>
            <div className="price-amount">
                {formattedPrice}
                {perNight && <span className="text-xs ml-1">/night</span>}
            </div>
            {isStale && (
                <span className="price-stale text-xs text-yellow-600" title={`Cached at ${data.cachedAt}`}>
                    prices may be outdated
                </span>
            )}
        </div>
    );
}
