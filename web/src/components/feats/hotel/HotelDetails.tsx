'use client';

import { useState, useEffect } from 'react';
import type { Hotel } from '@/lib/hotel-utils';
import { getHotelImageUrl } from '@/lib/hotel-utils';
import PriceDisplay from '@/components/ui/PriceDisplay';

interface HotelDetailsProps {
  product: Hotel;
  onAddToCart: () => void;
  addedToCart: boolean;
}

const PriceTotalDisplay = ({ productId, nights }: { productId: string; nights: number }) => {
  const [price, setPrice] = useState<number | null>(null);

  useEffect(() => {
    const fetchPrice = async () => {
      try {
        const sessionRes = await fetch('/api/session');
        const sessionData = await sessionRes.json();
        const params = new URLSearchParams({
          productId,
          sessionId: sessionData.sessionId || '',
          experimentId: sessionData.experimentId || '',
        });
        const res = await fetch(`/api/pricing?${params.toString()}`);
        const data = await res.json();
        setPrice(data.price);
      } catch (err) {
        console.error('failed to fetch price for total:', err);
      }
    };
    fetchPrice();
  }, [productId]);

  if (!price) return <span className="text-4xl font-bold text-gray-900">Loading...</span>;

  return (
    <span className="text-4xl font-bold text-gray-900">
      ${(price * nights).toFixed(2)}
    </span>
  );
};

export default function HotelDetails({ product, onAddToCart, addedToCart }: HotelDetailsProps) {
  return (
    <div className="w-full flex flex-col lg:flex-row gap-12 py-8">
      <div className="w-full lg:w-1/2 rounded-lg aspect-[4/3] overflow-hidden shrink-0">
        <img
          src={getHotelImageUrl(product.id, { w: 800, h: 600 })}
          alt={product.name}
          className="w-full h-full object-cover"
          onError={(e) => {
            e.currentTarget.style.display = 'none';
            if (e.currentTarget.nextElementSibling) {
              (e.currentTarget.nextElementSibling as HTMLElement).style.display = 'flex';
            }
          }}
        />
        <div className="w-full h-full bg-gray-100 rounded-lg flex items-center justify-center" style={{ display: 'none' }}>
          <span className="text-gray-400 text-lg font-medium">Hotel Image</span>
        </div>
      </div>

      <div className="flex-1 flex flex-col">
        <div className="border-b pb-6 mb-6">
          <h1 className="text-4xl font-bold text-gray-900 mb-2">{product.name}</h1>
        </div>

        <div className="grid grid-cols-2 gap-8 mb-8">
          <div>
            <h3 className="text-sm font-semibold text-gray-900 uppercase tracking-wider mb-2">Check-in</h3>
            <p className="text-lg text-gray-700">{product.checkIn}</p>
          </div>
          <div>
            <h3 className="text-sm font-semibold text-gray-900 uppercase tracking-wider mb-2">Check-out</h3>
            <p className="text-lg text-gray-700">{product.checkOut}</p>
          </div>
        </div>

        <div className="mb-8">
          <h3 className="text-sm font-semibold text-gray-900 uppercase tracking-wider mb-3">Amenities</h3>
          <div className="flex flex-wrap gap-3">
            {product.amenities.map(a => (
              <span key={a} className="px-3 py-1.5 bg-gray-100 text-gray-700 rounded-md text-sm font-medium">
                  {a.replaceAll('_', ' ')}
              </span>
            ))}
          </div>
        </div>

        <div className="mt-auto pt-6 border-t flex items-center justify-between">
          <div>
            <p className="text-sm text-gray-500 mb-1">Price per night</p>
            <div className="mb-3">
              <PriceDisplay productId={product.id} className="!text-2xl" />
            </div>
          </div>

          <button
            onClick={onAddToCart}
            disabled={addedToCart}
            className="px-8 py-4 bg-black hover:bg-gray-800 disabled:bg-green-600 text-white rounded-lg text-lg font-medium transition-all min-w-[200px]"
          >
            {addedToCart ? 'In Cart' : 'Add to Cart'}
          </button>
        </div>
      </div>
    </div>
  );
}
