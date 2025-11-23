'use client';

import type { EventName } from '@/lib/events';
import type { Hotel } from '@/lib/hotel-utils';
import { useHoverTracking } from '@/hooks/useHoverTracking';
import PriceDisplay from '@/components/ui/PriceDisplay';

const dispatchInteraction = (eventName: EventName, productId?: string, metadata?: Record<string, unknown>) => {
    const e = new CustomEvent('definedInteraction', {
        detail: { eventName, productId, metadata },
    });
    document.dispatchEvent(e);
};

const AmenityIcon = ({ name }: { name: string }) => {
    const iconMap: Record<string, string> = {
        wifi: 'Wi-Fi',
        pool: 'Pool',
        gym: 'Gym',
        parking: 'Parking',
        breakfast: 'Breakfast',
        spa: 'Spa',
    };
    return <span className="feature-tag">{iconMap[name.toLowerCase()] || name}</span>;
};

export default function HotelCard({ hotel }: { hotel: Hotel }) {
    const titleRef = useHoverTracking({
        eventName: 'hover_over_title',
        productId: hotel.id,
        metadata: { elementText: hotel.name, dateIndex: hotel.dateIndex },
    });

    const priceRef = useHoverTracking({
        eventName: 'hover_over_paragraph',
        productId: hotel.id,
        metadata: { elementText: 'price', dateIndex: hotel.dateIndex },
    });

    const handleCardClick = () => {
        dispatchInteraction('view_item_page', hotel.id, {
            roomType: hotel.roomType,
            price: hotel.pricePerNight,
            nights: hotel.nights,
            dateIndex: hotel.dateIndex,
        });
    };

    return (
        <div
            className="hotel-card cursor-pointer"
            onClick={handleCardClick}
        >
            <div className="hotel-image bg-gray-200 flex items-center justify-center">
                <span className="text-gray-400 text-sm">Image</span>
            </div>

            <div className="hotel-info">
                <h3 ref={titleRef} className="hotel-name">{hotel.name}</h3>
                <div className="hotel-location text-sm mb-2">{hotel.roomType}</div>
                <div className="text-sm text-[var(--text-secondary)] mb-2">
                    {hotel.checkIn} - {hotel.checkOut}
                </div>
                <div className="hotel-features">
                    {hotel.amenities.map((a) => (
                        <AmenityIcon key={a} name={a} />
                    ))}
                </div>
                {hotel.refundable && (
                    <div className="free-cancellation mt-2">Free cancellation</div>
                )}
            </div>

            <div className="hotel-pricing">
                <div ref={priceRef}>
                    <PriceDisplay
                        productId={hotel.id}
                        className="price-wrapper"
                        perNight
                    />
                </div>
                <div className="text-xs text-[var(--text-secondary)] mt-1">
                    Total for {hotel.nights} night{hotel.nights > 1 ? 's' : ''}
                </div>
            </div>
        </div>
    );
}
