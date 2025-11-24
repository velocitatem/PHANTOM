'use client';

import { useState, useEffect, Suspense } from 'react';
import { useSearchParams } from 'next/navigation';
import { Navigation } from '@/components/ui';
import HotelCard from '@/components/feats/hotel/HotelCard';
import { transformProduct, type Hotel, type HotelProduct } from '@/lib/hotel-utils';

function RoomsList() {
  const searchParams = useSearchParams();
  const [rooms, setRooms] = useState<Hotel[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchRooms = async () => {
      try {
        const url = new URL('/api/products', window.location.origin);
        url.searchParams.set('type', 'hotel');

        // forward all relevant search params to the API
        const params = ['dateIndex', 'destination', 'adults', 'rooms'];
        params.forEach(param => {
          const val = searchParams.get(param);
          if (val) url.searchParams.set(param, val);
        });

        const res = await fetch(url.toString());
        if (!res.ok) throw new Error(`Failed to fetch: ${res.status}`);
        const json = await res.json();
        const transformed = json.data.map((p: HotelProduct) => transformProduct(p));
        setRooms(transformed);
      } catch (e) {
        setError(e instanceof Error ? e.message : 'Failed to load products');
        console.error('[FETCH_ERROR]', e);
      } finally {
        setLoading(false);
      }
    };
    fetchRooms();
  }, [searchParams]);

  return (
    <>
      <h1 className="text-3xl font-bold mb-6">Available Rooms</h1>
      {loading && <div className="text-center py-8">Loading...</div>}
      {error && <div className="text-red-500 text-center py-8">{error}</div>}
      {!loading && !error && (
        <div className="space-y-4">
          {rooms.map((r) => (
            <HotelCard key={r.id} hotel={r} />
          ))}
        </div>
      )}
    </>
  );
}

export default function HotelProducts() {
  return (
    <>
      <Navigation />
      <main className="max-w-7xl mx-auto px-4 py-8">
        <Suspense fallback={<div className="text-center py-8">Loading...</div>}>
          <RoomsList />
        </Suspense>
      </main>
    </>
  );
}
