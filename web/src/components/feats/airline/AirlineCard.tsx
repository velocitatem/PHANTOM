'use client';

import type { EventName } from '@/lib/events';

const dispatchInteraction = (eventName: EventName, productId?: string, metadata?: Record<string, unknown>) => {
  const e = new CustomEvent('definedInteraction', {
    detail: { eventName, productId, metadata },
  });
  document.dispatchEvent(e);
};

type CabinClass = 'economy' | 'premium' | 'business' | 'first';
type FareRule = 'flexible' | 'standard' | 'basic';

interface Flight {
  id: string;
  departure: { time: string; airport: string };
  arrival: { time: string; airport: string };
  duration: string;
  stops: number;
  cabinClass: CabinClass;
  fareRule: FareRule;
  refundable: boolean;
  basePrice: number;
}

const PriceDisplay = ({ price }: { price: number }) => (
  <div className="fare-price">${price}</div>
);

export default function AirlineCard({ flight }: { flight: Flight }) {
  const handleCardClick = () => {
    dispatchInteraction('product_view', flight.id, {
      cabinClass: flight.cabinClass,
      fareRule: flight.fareRule,
      price: flight.basePrice,
    });
  };

  return (
    <div
      className="flight-card cursor-pointer"
      onClick={handleCardClick}
      onMouseEnter={() => dispatchInteraction('product_hover', flight.id)}
    >
      <div className="flight-timing">
        <div className="flight-time">{flight.departure.time}</div>
        <div className="flight-airport">{flight.departure.airport}</div>
      </div>

      <div className="flight-route">
        <div className="flight-duration">{flight.duration}</div>
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
        <PriceDisplay price={flight.basePrice} />
      </div>
    </div>
  );
}
