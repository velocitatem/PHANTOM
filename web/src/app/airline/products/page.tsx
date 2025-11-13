'use client';

import { Navigation } from '@/components/ui';
import AirlineCard from '@/components/feats/airline/AirlineCard';

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

const genRandomFlights = (): Flight[] => {
  const airports = ['JFK', 'LAX', 'ORD', 'ATL', 'DFW', 'SFO', 'SEA', 'MIA'];
  const cabins: CabinClass[] = ['economy', 'premium', 'business', 'first'];
  const fareRules: FareRule[] = ['flexible', 'standard', 'basic'];

  return Array.from({ length: 12 }, (_, i) => {
    const depHour = Math.floor(Math.random() * 24);
    const arrHour = (depHour + Math.floor(Math.random() * 6) + 2) % 24;
    const stops = Math.random() > 0.6 ? 0 : Math.floor(Math.random() * 2) + 1;
    const cabin = cabins[Math.floor(Math.random() * cabins.length)];
    const fareRule = fareRules[Math.floor(Math.random() * fareRules.length)];

    const basePrice = Math.floor(
      (cabin === 'economy' ? 200 : cabin === 'premium' ? 400 : cabin === 'business' ? 800 : 1500) +
      Math.random() * 300
    );

    return {
      id: `flt-${i}`,
      departure: {
        time: `${depHour.toString().padStart(2, '0')}:${Math.floor(Math.random() * 60).toString().padStart(2, '0')}`,
        airport: airports[Math.floor(Math.random() * airports.length)],
      },
      arrival: {
        time: `${arrHour.toString().padStart(2, '0')}:${Math.floor(Math.random() * 60).toString().padStart(2, '0')}`,
        airport: airports[Math.floor(Math.random() * airports.length)],
      },
      duration: `${Math.floor(Math.random() * 5) + 2}h ${Math.floor(Math.random() * 60)}m`,
      stops,
      cabinClass: cabin,
      fareRule,
      refundable: Math.random() > 0.7,
      basePrice,
    };
  });
};

export default function AirlineProducts() {
  const flights = genRandomFlights();

  return (
    <>
      <Navigation />
      <main className="max-w-7xl mx-auto px-4 py-8">
        <h1 className="text-3xl font-bold mb-6">Available Flights</h1>
        <div className="space-y-4">
          {flights.map((f) => (
            <AirlineCard key={f.id} flight={f} />
          ))}
        </div>
      </main>
    </>
  );
}
