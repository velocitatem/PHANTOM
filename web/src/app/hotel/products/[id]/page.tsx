'use client';

import { useState, useEffect } from 'react';
import { useParams, useRouter } from 'next/navigation';
import { Navigation } from '@/components/ui';
import { useCart } from '@/contexts/CartContext';
import HotelDetails from '@/components/feats/hotel/HotelDetails';
import { transformProduct, type Hotel, type HotelProduct } from '@/lib/hotel-utils';
import type { EventName } from '@/lib/events';

const dispatchInteraction = (eventName: EventName, productId?: string, metadata?: Record<string, unknown>) => {
    const e = new CustomEvent('definedInteraction', {
        detail: { eventName, productId, metadata },
    });
    document.dispatchEvent(e);
};

export default function HotelProductPage() {
    const params = useParams();
    const router = useRouter();
    const { addItem } = useCart();
    const [product, setProduct] = useState<Hotel | null>(null);
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
                const transformed = transformProduct(json.data as HotelProduct);
                setProduct(transformed);

                // fire learn_more_about_item event when product loads
                dispatchInteraction('learn_more_about_item', productId, {
                    type: 'hotel',
                    dateIndex: transformed.dateIndex,
                    roomType: transformed.roomType,
                });
            } catch (e) {
                setError(e instanceof Error ? e.message : 'Failed to load product');
                console.error('[FETCH_HOTEL_ERROR]', e);
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
            type: 'hotel',
            name: product.name,
            price: product.pricePerNight,
            metadata: {
                roomType: product.roomType,
                nights: product.nights,
                checkIn: product.checkIn,
                checkOut: product.checkOut,
            },
            dateIndex: product.dateIndex,
        });

        dispatchInteraction('add_item_to_cart', productId, {
            type: 'hotel',
            price: product.pricePerNight,
        });

        setAdded(true);
        setTimeout(() => setAdded(false), 2000);
    };

    return (
        <>
            <Navigation />
            <main className="max-w-4xl mx-auto px-4 py-8">
                {loading && <div className="text-center py-8">Loading hotel details...</div>}
                {error && <div className="text-red-500 text-center py-8">{error}</div>}

                {!loading && !error && product && (
                    <>
                        <button
                            onClick={() => router.back()}
                            className="mt-6 text-blue-600 hover:underline"
                        >
                            ← Back to rooms
                        </button>
                        <HotelDetails
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
