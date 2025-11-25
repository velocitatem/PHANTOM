'use client';

import type { EventName } from '@/lib/events';
import type { Flight } from '@/lib/airline-utils';
import { useHoverTracking } from '@/hooks/useHoverTracking';
import PriceDisplay from '@/components/ui/PriceDisplay';

const dispatchInteraction = (eventName: EventName, productId?: string, metadata?: Record<string, unknown>) => {
    const e = new CustomEvent('definedInteraction', {
        detail: { eventName, productId, metadata },
    });
    document.dispatchEvent(e);
};

export default function AirlineCard({ flight }: { flight: Flight }) {
    const durationRef = useHoverTracking({
        eventName: 'hover_over_title',
        productId: flight.id,
        metadata: { elementText: flight.duration, dateIndex: flight.dateIndex },
    });

    const priceRef = useHoverTracking({
        eventName: 'hover_over_paragraph',
        productId: flight.id,
        metadata: { elementText: 'price', dateIndex: flight.dateIndex },
    });

    const handleCardClick = () => {
        dispatchInteraction('view_item_page', flight.id, {
            cabinClass: flight.cabinClass,
            fareRule: flight.fareRule,
            price: flight.basePrice,
            dateIndex: flight.dateIndex,
        });
        window.location.href = `/airline/products/${flight.id}`;
    };

    return (
        <div
            className="flight-card cursor-pointer"
            onClick={handleCardClick}
        >
            <div className="flight-timing">
                <div className="flight-time">{flight.departure.time}</div>
                <div className="flight-airport">{flight.departure.airport}</div>
            </div>

            <div className="flight-route">
                <div ref={durationRef} className="flight-duration">{flight.duration}</div>
                <div className="flight-stops">
                    {flight.stops === 0 ? 'Direct' : `${flight.stops} stop${flight.stops > 1 ? 's' : ''}`}
                </div>
            </div>

            <div className="flight-timing">
                <div className="flight-time">{flight.arrival.time}</div>
                <div className="flight-airport">{flight.arrival.airport}</div>
            </div>

            <div className="flight-pricing">
                <div className="fare-class capitalize mb-2">{flight.cabinClass}</div>
                <div className="text-sm text-[var(--text-secondary)] mb-2 capitalize">{flight.fareRule}</div>
                {flight.refundable && (
                    <div className="badge-value text-xs mb-2">Refundable</div>
                )}
                <div ref={priceRef}>
                    <PriceDisplay
                        productId={flight.id}
                        className="fare-price"
                    />
                </div>
            </div>
        </div>
    );
}
