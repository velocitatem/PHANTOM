'use client';

import type { Hotel } from '@/lib/hotel-utils';

interface HotelDetailsProps {
  product: Hotel;
  onAddToCart: () => void;
  addedToCart: boolean;
}

export default function HotelDetails({ product, onAddToCart, addedToCart }: HotelDetailsProps) {
  return (
    <div className="w-full flex flex-col lg:flex-row gap-12 py-8">
      {/* Image Section - Larger and cleaner */}
      <div className="w-full lg:w-1/2 bg-gray-100 rounded-lg aspect-[4/3] flex items-center justify-center shrink-0">
        <span className="text-gray-400 text-lg font-medium">Hotel Image</span>
      </div>

      {/* Details Section - Full height/width usage */}
      <div className="flex-1 flex flex-col">
        <div className="border-b pb-6 mb-6">
          <h1 className="text-4xl font-bold text-gray-900 mb-2">{product.name}</h1>
          <p className="text-xl text-gray-500">{product.roomType}</p>
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
                {a}
              </span>
            ))}
          </div>
        </div>

        {product.refundable && (
          <div className="mb-8 p-4 bg-green-50 text-green-800 rounded-md inline-block">
            <span className="font-medium">Free cancellation available</span>
          </div>
        )}

        <div className="mt-auto pt-6 border-t flex items-center justify-between">
          <div>
            <p className="text-sm text-gray-500 mb-1">Total for {product.nights} night{product.nights > 1 ? 's' : ''}</p>
            <div className="flex items-baseline gap-2">
              <span className="text-4xl font-bold text-gray-900">${product.pricePerNight * product.nights}</span>
              <span className="text-gray-500">/ {product.nights} nights</span>
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
