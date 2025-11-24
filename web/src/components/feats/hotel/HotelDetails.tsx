'use client';

import type { Hotel } from '@/lib/hotel-utils';

interface HotelDetailsProps {
  product: Hotel;
  onAddToCart: () => void;
  addedToCart: boolean;
}

export default function HotelDetails({ product, onAddToCart, addedToCart }: HotelDetailsProps) {
  return (
    <div className="space-y-6">
      <div className="bg-gray-100 h-64 flex items-center justify-center rounded-lg">
        <span className="text-gray-400">Hotel Image</span>
      </div>

      <div>
        <h1 className="text-3xl font-bold mb-2">{product.name}</h1>
        <p className="text-lg text-gray-600">{product.roomType}</p>
      </div>

      <div className="grid grid-cols-2 gap-4">
        <div>
          <h3 className="font-semibold mb-1">Check-in</h3>
          <p>{product.checkIn}</p>
        </div>
        <div>
          <h3 className="font-semibold mb-1">Check-out</h3>
          <p>{product.checkOut}</p>
        </div>
      </div>

      <div>
        <h3 className="font-semibold mb-2">Amenities</h3>
        <div className="flex flex-wrap gap-2">
          {product.amenities.map(a => (
            <span key={a} className="px-3 py-1 bg-blue-100 text-blue-800 rounded-full text-sm">
              {a}
            </span>
          ))}
        </div>
      </div>

      {product.refundable && (
        <div className="bg-green-50 border border-green-200 rounded-lg p-4">
          <p className="text-green-800 font-medium">Free cancellation available</p>
        </div>
      )}

      <div className="border-t pt-4">
        <div className="flex justify-between items-center">
          <div>
            <p className="text-sm text-gray-600">Price per night</p>
            <p className="text-3xl font-bold">${product.pricePerNight}</p>
            <p className="text-sm text-gray-500">
              Total for {product.nights} night{product.nights > 1 ? 's' : ''}: ${product.pricePerNight * product.nights}
            </p>
          </div>
        </div>
      </div>

      <div className="flex gap-4">
        <button
          onClick={onAddToCart}
          disabled={addedToCart}
          className="w-full px-6 py-3 bg-blue-600 hover:bg-blue-700 disabled:bg-green-600 text-white rounded-lg font-medium transition-colors"
        >
          {addedToCart ? 'Added to Cart!' : 'Add to Cart'}
        </button>
      </div>
    </div>
  );
}
