'use client';

import { useState, useEffect, Suspense } from 'react';
import { useSearchParams } from 'next/navigation';
import { Navigation } from '@/components/ui';
import AirlineCard from '@/components/feats/airline/AirlineCard';
import { transformProduct, type Flight, type AirlineProduct } from '@/lib/airline-utils';

function FlightsList() {
  const searchParams = useSearchParams();
  const [flights, setFlights] = useState<Flight[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchFlights = async () => {
      try {
        const url = new URL('/api/products', window.location.origin);
        url.searchParams.set('type', 'airline');

        // forward all relevant search params to the API
        const params = ['dateIndex', 'origin', 'destination', 'tripType', 'adults', 'children', 'infants'];
        params.forEach(param => {
          const val = searchParams.get(param);
          if (val) url.searchParams.set(param, val);
        });

        const res = await fetch(url.toString());
        if (!res.ok) throw new Error(`Failed to fetch: ${res.status}`);
        const json = await res.json();
        const transformed = json.data.map((p: AirlineProduct) => transformProduct(p));
        setFlights(transformed);
      } catch (e) {
        setError(e instanceof Error ? e.message : 'Failed to load products');
        console.error('[FETCH_ERROR]', e);
      } finally {
        setLoading(false);
      }
    };
    fetchFlights();
  }, [searchParams]);

  return (
    <>
      <h1 className="text-3xl font-bold mb-6">Available Flights</h1>
      {loading && <div className="text-center py-8">Loading...</div>}
      {error && <div className="text-red-500 text-center py-8">{error}</div>}
      {!loading && !error && (
        <div className="space-y-4">
          {flights.map((f) => (
            <AirlineCard key={f.id} flight={f} />
          ))}
        </div>
      )}
    </>
  );
}

export default function AirlineProducts() {
  return (
    <>
      <Navigation />
      <main className="max-w-7xl mx-auto px-4 py-8">
        <Suspense fallback={<div className="text-center py-8">Loading...</div>}>
          <FlightsList />
        </Suspense>
      </main>
    </>
  );
}
