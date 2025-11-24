'use client';

import type { Flight } from '@/lib/airline-utils';

interface AirlineDetailsProps {
  product: Flight;
  onAddToCart: () => void;
  addedToCart: boolean;
}

export default function AirlineDetails({ product, onAddToCart, addedToCart }: AirlineDetailsProps) {
  return (
    <div className="space-y-6">
      <div className="bg-gray-100 h-64 flex items-center justify-center rounded-lg">
        <span className="text-gray-400">Flight Image</span>
      </div>

      <div>
        <h1 className="text-3xl font-bold mb-2">{product.flightType}</h1>
        <p className="text-lg text-gray-600">{product.cabinClass} Class</p>
      </div>

      <div className="grid grid-cols-2 gap-6">
        <div>
          <h3 className="font-semibold mb-2">Departure</h3>
          <p className="text-xl font-bold">{product.departure.time}</p>
          <p className="text-gray-600">{product.departure.airport}</p>
        </div>
        <div>
          <h3 className="font-semibold mb-2">Arrival</h3>
          <p className="text-xl font-bold">{product.arrival.time}</p>
          <p className="text-gray-600">{product.arrival.airport}</p>
        </div>
      </div>

      <div className="grid grid-cols-3 gap-4 py-4 border-y">
        <div>
          <p className="text-sm text-gray-600">Duration</p>
          <p className="font-semibold">{product.duration}</p>
        </div>
        <div>
          <p className="text-sm text-gray-600">Stops</p>
          <p className="font-semibold">
            {product.stops === 0 ? 'Nonstop' : `${product.stops} stop${product.stops > 1 ? 's' : ''}`}
          </p>
        </div>
        <div>
          <p className="text-sm text-gray-600">Availability</p>
          <p className="font-semibold">{product.availability} seats</p>
        </div>
      </div>

      <div>
        <h3 className="font-semibold mb-2">Fare Details</h3>
        <p className="text-sm text-gray-600 mb-1">{product.fareRule}</p>
        {product.refundable && (
          <div className="bg-green-50 border border-green-200 rounded-lg p-3 mt-2">
            <p className="text-green-800 text-sm font-medium">Refundable fare</p>
          </div>
        )}
      </div>

      <div className="border-t pt-4">
        <div className="flex justify-between items-center">
          <div>
            <p className="text-sm text-gray-600">Total price</p>
            <p className="text-3xl font-bold">${product.basePrice}</p>
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
