'use client';

import { useState, FormEvent } from 'react';
import { Button, Label, Input, DateInput, Dropdown, DropdownCounter } from '@/components/ui';

const LocationIcon = () => (
  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17.657 16.657L13.414 20.9a1.998 1.998 0 01-2.827 0l-4.244-4.243a8 8 0 1111.314 0z" />
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 11a3 3 0 11-6 0 3 3 0 016 0z" />
  </svg>
);

export default function HotelHero() {
  const [destination, setDestination] = useState('');
  const [checkIn, setCheckIn] = useState('');
  const [checkOut, setCheckOut] = useState('');
  const [guests, setGuests] = useState({ adults: 2, rooms: 1 });

  const handleSearch = (e: FormEvent) => {
    e.preventDefault();
    console.log({ destination, checkIn, checkOut, guests });
  };

  return (
    <div className="hero-section min-h-[70vh] flex items-center justify-center">
      <div className="w-full max-w-4xl px-4">
        <div className="text-center mb-8">
          <h1 className="text-4xl md:text-5xl font-bold mb-4">
            Find your perfect stay
          </h1>
          <p className="text-lg">
            Search hotels, compare prices, and book with confidence
          </p>
        </div>

        <form onSubmit={handleSearch} className="search-form">
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
            <div className="sm:col-span-2">
              <Label htmlFor="destination">Where to?</Label>
              <Input
                type="text"
                id="destination"
                value={destination}
                onChange={(e) => setDestination(e.target.value)}
                placeholder="City, hotel, or landmark"
                icon={<LocationIcon />}
                required
              />
            </div>

            <div>
              <Label htmlFor="checkIn">Check-in</Label>
              <DateInput
                id="checkIn"
                value={checkIn}
                onChange={(e) => setCheckIn(e.target.value)}
                required
              />
            </div>

            <div>
              <Label htmlFor="checkOut">Check-out</Label>
              <DateInput
                id="checkOut"
                value={checkOut}
                onChange={(e) => setCheckOut(e.target.value)}
                required
              />
            </div>

            <div className="sm:col-span-2 lg:col-span-4">
              <Label htmlFor="guests">Guests & Rooms</Label>
              <Dropdown label={`${guests.adults} ${guests.adults === 1 ? 'adult' : 'adults'}, ${guests.rooms} ${guests.rooms === 1 ? 'room' : 'rooms'}`}>
                <DropdownCounter
                  label="Adults"
                  value={guests.adults}
                  min={1}
                  onChange={(v) => setGuests({ ...guests, adults: v })}
                />
                <DropdownCounter
                  label="Rooms"
                  value={guests.rooms}
                  min={1}
                  onChange={(v) => setGuests({ ...guests, rooms: v })}
                />
              </Dropdown>
            </div>

            <div className="sm:col-span-2 lg:col-span-4">
              <Button type="submit" fullWidth>
                Search Hotels
              </Button>
            </div>
          </div>
        </form>

        <div className="mt-6 text-center text-sm">
          <p>Over 2 million hotels worldwide · Best price guarantee · Free cancellation on most bookings</p>
        </div>
      </div>
    </div>
  );
}
