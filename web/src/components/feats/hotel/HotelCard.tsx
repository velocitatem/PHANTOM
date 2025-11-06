'use client';

import type { EventName } from '@/lib/events';

const dispatchInteraction = (eventName: EventName, productId?: string, metadata?: Record<string, unknown>) => {
  const e = new CustomEvent('definedInteraction', {
    detail: { eventName, productId, metadata },
  });
  document.dispatchEvent(e);
};

interface Hotel {
  id: string;
  name: string;
  roomType: string;
  checkIn: string;
  checkOut: string;
  amenities: string[];
  refundable: boolean;
  pricePerNight: number;
  nights: number;
}

const PriceDisplay = ({ price, perNight }: { price: number; perNight: boolean }) => (
  <div className="price-wrapper">
    <div className="price-label">{perNight ? 'Per night' : 'Total'}</div>
    <div className="price-amount">${price}</div>
    {perNight && <div className="price-unit">/night</div>}
  </div>
);

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
  const handleCardClick = () => {
    dispatchInteraction('product_view', hotel.id, {
      roomType: hotel.roomType,
      price: hotel.pricePerNight,
      nights: hotel.nights,
    });
  };

  return (
    <div
      className="hotel-card cursor-pointer"
      onClick={handleCardClick}
      onMouseEnter={() => dispatchInteraction('product_hover', hotel.id)}
    >
      <div className="hotel-image bg-gray-200 flex items-center justify-center">
        <span className="text-gray-400 text-sm">Image</span>
      </div>

      <div className="hotel-info">
        <h3 className="hotel-name">{hotel.name}</h3>
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
        <PriceDisplay price={hotel.pricePerNight} perNight />
        <div className="text-xs text-[var(--text-secondary)] mt-1">
          ${hotel.pricePerNight * hotel.nights} total for {hotel.nights} night{hotel.nights > 1 ? 's' : ''}
        </div>
      </div>
    </div>
  );
}
