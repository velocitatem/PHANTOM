'use client';

import type { Flight } from '@/lib/airline-utils';

interface AirlineDetailsProps {
  product: Flight;
  onAddToCart: () => void;
  addedToCart: boolean;
}

export default function AirlineDetails({ product, onAddToCart, addedToCart }: AirlineDetailsProps) {
  return (
    <div className="w-full flex flex-col lg:flex-row gap-12 py-8">
      {/* Image Section */}
      <div className="w-full lg:w-1/3 bg-gray-100 rounded-lg aspect-square flex items-center justify-center shrink-0">
        <span className="text-gray-400 text-lg font-medium">Flight Image</span>
      </div>

      {/* Details Section */}
      <div className="flex-1 flex flex-col">
        <div className="flex justify-between items-start border-b pb-6 mb-6">
          <div>
            <h1 className="text-3xl font-bold text-gray-900 mb-1">{product.flightType}</h1>
            <p className="text-lg text-gray-500">{product.cabinClass} Class</p>
          </div>
          <div className="text-right">
            <p className="text-4xl font-bold text-gray-900">${product.basePrice}</p>
            {product.refundable && (
              <span className="inline-block mt-2 px-3 py-1 bg-green-50 text-green-700 rounded-full text-xs font-medium">
                Refundable
              </span>
            )}
          </div>
        </div>

        <div className="flex items-center justify-between mb-10">
          <div className="text-center min-w-[100px]">
            <p className="text-3xl font-bold text-gray-900">{product.departure.time}</p>
            <p className="text-sm text-gray-500 font-medium mt-1">{product.departure.airport}</p>
          </div>

          <div className="flex-1 px-8 flex flex-col items-center">
            <p className="text-sm text-gray-500 mb-2">{product.duration}</p>
            <div className="w-full h-0.5 bg-gray-200 relative flex items-center justify-center">
              <div className="absolute w-3 h-3 bg-gray-400 rounded-full"></div>
            </div>
            <p className="text-sm text-gray-500 mt-2">
              {product.stops === 0 ? 'Nonstop' : `${product.stops} stop${product.stops > 1 ? 's' : ''}`}
            </p>
          </div>

          <div className="text-center min-w-[100px]">
            <p className="text-3xl font-bold text-gray-900">{product.arrival.time}</p>
            <p className="text-sm text-gray-500 font-medium mt-1">{product.arrival.airport}</p>
          </div>
        </div>

        <div className="mt-auto flex items-center justify-between pt-6 border-t">
          <div className="text-gray-600">
            <span className="font-bold text-gray-900">{product.availability}</span> seats remaining
            <span className="mx-2">•</span>
            {product.fareRule}
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
