'use client';

import { useState, useEffect } from 'react';
import { useParams, useRouter } from 'next/navigation';
import { Navigation } from '@/components/ui';
import { useCart } from '@/contexts/CartContext';
import AirlineDetails from '@/components/feats/airline/AirlineDetails';
import { transformProduct, type Flight, type AirlineProduct } from '@/lib/airline-utils';
import type { EventName } from '@/lib/events';

const dispatchInteraction = (eventName: EventName, productId?: string, metadata?: Record<string, unknown>) => {
    const e = new CustomEvent('definedInteraction', {
        detail: { eventName, productId, metadata },
    });
    document.dispatchEvent(e);
};

export default function AirlineProductPage() {
    const params = useParams();
    const router = useRouter();
    const { addItem } = useCart();
    const [product, setProduct] = useState<Flight | null>(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);
    const [added, setAdded] = useState(false);

    const productId = params.id as string;

    useEffect(() => {
        const fetchProduct = async () => {
            try {
                const res = await fetch(`/api/products/${productId}`);
                if (!res.ok) throw new Error(`Failed to fetch: ${res.status}`);
                const json = await res.json();
                const transformed = transformProduct(json.data as AirlineProduct);
                setProduct(transformed);

                // fire learn_more_about_item event when product loads
                dispatchInteraction('learn_more_about_item', productId, {
                    type: 'airline',
                    dateIndex: transformed.dateIndex,
                    flightType: transformed.flightType,
                });
            } catch (e) {
                setError(e instanceof Error ? e.message : 'Failed to load product');
                console.error('[FETCH_FLIGHT_ERROR]', e);
            } finally {
                setLoading(false);
            }
        };
        fetchProduct();
    }, [productId]);

    const handleAddToCart = () => {
        if (!product) return;

        addItem({
            id: productId,
            type: 'airline',
            name: product.flightType,
            price: product.basePrice,
            metadata: {
                departure: product.departure,
                arrival: product.arrival,
                duration: product.duration,
                cabinClass: product.cabinClass,
            },
            dateIndex: product.dateIndex,
        });

        dispatchInteraction('add_item_to_cart', productId, {
            type: 'airline',
            price: product.basePrice,
        });

        setAdded(true);
        setTimeout(() => setAdded(false), 2000);
    };

    return (
        <>
            <Navigation />
            <main className="max-w-4xl mx-auto px-4 py-8">
                {loading && <div className="text-center py-8">Loading flight details...</div>}
                {error && <div className="text-red-500 text-center py-8">{error}</div>}

                {!loading && !error && product && (
                    <>
                        <button
                            onClick={() => router.back()}
                            className="mt-6 text-blue-600 hover:underline"
                        >
                            ← Back to flights
                        </button>
                        <AirlineDetails
                            product={product}
                            onAddToCart={handleAddToCart}
                            addedToCart={added}
                        />

                    </>
                )}
            </main>
        </>
    );
}
