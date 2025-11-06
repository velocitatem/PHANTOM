'use client';

import { Navigation } from '@/components/ui';
import HotelCard from '@/components/feats/hotel/HotelCard';

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

const genRandomHotels = (): Hotel[] => {
  const names = [
    'Grand Plaza Hotel',
    'Seaside Resort',
    'Downtown Suites',
    'Mountain View Lodge',
    'City Center Inn',
    'Luxury Beach Resort',
    'Urban Boutique Hotel',
    'Garden View Hotel',
  ];
  const roomTypes = ['Standard Room', 'Deluxe Room', 'Suite', 'Executive Suite', 'Premium Room'];
  const amenities = ['wifi', 'pool', 'gym', 'parking', 'breakfast', 'spa'];

  return Array.from({ length: 10 }, (_, i) => {
    const nights = Math.floor(Math.random() * 5) + 1;
    const basePrice = Math.floor(80 + Math.random() * 220);
    const selectedAmenities = amenities
      .sort(() => Math.random() - 0.5)
      .slice(0, Math.floor(Math.random() * 3) + 2);

    const today = new Date();
    const checkInDate = new Date(today);
    checkInDate.setDate(today.getDate() + Math.floor(Math.random() * 10));
    const checkOutDate = new Date(checkInDate);
    checkOutDate.setDate(checkInDate.getDate() + nights);

    return {
      id: `htl-${i}`,
      name: names[i % names.length],
      roomType: roomTypes[Math.floor(Math.random() * roomTypes.length)],
      checkIn: checkInDate.toLocaleDateString('en-US', { month: 'short', day: 'numeric' }),
      checkOut: checkOutDate.toLocaleDateString('en-US', { month: 'short', day: 'numeric' }),
      amenities: selectedAmenities,
      refundable: Math.random() > 0.5,
      pricePerNight: basePrice,
      nights,
    };
  });
};

export default function HotelProducts() {
  const hotels = genRandomHotels();

  return (
    <>
      <Navigation />
      <main className="max-w-7xl mx-auto px-4 py-8">
        <h1 className="text-3xl font-bold mb-6">Available Hotels</h1>
        <div className="space-y-4">
          {hotels.map((h) => (
            <HotelCard key={h.id} hotel={h} />
          ))}
        </div>
      </main>
    </>
  );
}
