'use client';

import { Navigation } from '@/components/ui';
import { useCart } from '@/contexts/CartContext';
import type { EventName } from '@/lib/events';

const dispatchInteraction = (eventName: EventName, productId?: string, metadata?: Record<string, unknown>) => {
    const e = new CustomEvent('definedInteraction', {
        detail: { eventName, productId, metadata },
    });
    document.dispatchEvent(e);
};

export default function CartPage() {
    const { items, removeItem, clearCart, itemCount } = useCart();

    const handleRemove = (id: string, type: string) => {
        removeItem(id);
        dispatchInteraction('remove_item', id, { type });
    };
    let itemTypes = Array.from(new Set(items.map(item => item.type)))[0] || 'items';


    const total = items.reduce((sum, item) => sum + item.price, 0);

    return (
        <>
            <Navigation />
            <main className="max-w-4xl mx-auto px-4 py-8">
                <div className="flex justify-between items-center mb-6">
                    <h1 className="text-3xl font-bold">Shopping Cart</h1>
                    {itemCount > 0 && (
                        <button
                            onClick={clearCart}
                            className="text-sm hover:underline"
                            style={{ color: 'var(--accent-warning)' }}
                        >
                            Clear cart
                        </button>
                    )}
                </div>

                {itemCount === 0 ? (
                    <div className="text-center py-12">
                        <p className="text-gray-500 mb-4">Your cart is empty</p>
                        <a href="/" className="hover:underline" style={{ color: 'var(--text-accent)' }}>Browse our selection</a>
                    </div>
                ) : (
                    <>
                        <div className="space-y-4 mb-8">
                            {items.map(item => (
                                <div
                                    key={item.id}
                                    className="flex justify-between items-start p-4 border rounded-lg hover:bg-gray-50"
                                >
                                    <div className="flex-1">
                                        <div className="flex items-center gap-2 mb-1">
                                            <h3 className="font-semibold">{item.name}</h3>
                                        </div>

                                        {item.type === 'hotel' && (
                                            <div className="text-sm text-gray-600">
                                                <p>{String(item.metadata.checkIn)} - {String(item.metadata.checkOut)}</p>
                                                <p>{String(item.metadata.nights)} night{Number(item.metadata.nights) > 1 ? 's' : ''}</p>
                                            </div>
                                        )}

                                        {item.type === 'airline' && (
                                            <div className="text-sm text-gray-600">
                                                <p>{String(item.metadata.cabinClass)} Class</p>
                                                <p>{String((item.metadata.departure as any)?.airport)} → {String((item.metadata.arrival as any)?.airport)}</p>
                                                <p>Duration: {String(item.metadata.duration)}</p>
                                            </div>
                                        )}
                                    </div>

                                    <div className="text-right ml-4">
                                        <p className="text-xl font-bold mb-2">${item.price}</p>
                                        <button
                                            onClick={() => handleRemove(item.id, item.type)}
                                            className="text-sm hover:underline"
                                            style={{ color: 'var(--accent-warning)' }}
                                        >
                                            Remove
                                        </button>
                                    </div>
                                </div>
                            ))}
                        </div>

                        <div className="border-t pt-4">
                            <div className="flex justify-between items-center mb-4">
                                <span className="text-xl font-semibold">Total</span>
                                <span className="text-3xl font-bold">${total.toFixed(2)}</span>
                            </div>
                            <button
                                onClick={() => {
                                    dispatchInteraction('checkout_start', undefined, { total, itemCount });
                                    window.location.href = '/checkout';
                                }}
                                className="btn-primary w-full"
                            >
                                Proceed to Checkout
                            </button>
                        </div>
                    </>
                )}
            </main>
        </>
    );
}
