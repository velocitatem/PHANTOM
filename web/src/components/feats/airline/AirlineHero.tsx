'use client';

import { useState, FormEvent } from 'react';
import { Button, Label, Input, DateInput, RadioGroup, Dropdown, DropdownCounter } from '@/components/ui';

type TripType = 'roundtrip' | 'oneway' | 'multicity';

const PlaneIcon = () => (
  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8" />
  </svg>
);

const LocationIcon = () => (
  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17.657 16.657L13.414 20.9a1.998 1.998 0 01-2.827 0l-4.244-4.243a8 8 0 1111.314 0z" />
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 11a3 3 0 11-6 0 3 3 0 016 0z" />
  </svg>
);

export default function AirlineHero() {
  const [tripType, setTripType] = useState<TripType>('roundtrip');
  const [origin, setOrigin] = useState('');
  const [destination, setDestination] = useState('');
  const [departDate, setDepartDate] = useState('');
  const [returnDate, setReturnDate] = useState('');
  const [passengers, setPassengers] = useState({ adults: 1, children: 0, infants: 0 });

  const handleSearch = (e: FormEvent) => {
    e.preventDefault();
    console.log({ tripType, origin, destination, departDate, returnDate, passengers });
  };

  const totalPax = passengers.adults + passengers.children + passengers.infants;

  return (
    <div className="hero-section min-h-[70vh] flex items-center justify-center">
      <div className="w-full max-w-5xl px-4">
        <div className="text-center mb-8">
          <h1 className="text-4xl md:text-5xl font-bold mb-4">
            Book flights at the best prices
          </h1>
          <p className="text-lg">
            Compare hundreds of airlines and find the perfect flight for your journey
          </p>
        </div>

        <div className="search-form">
          <form onSubmit={handleSearch}>
            <div className="mb-6">
              <RadioGroup
                name="tripType"
                value={tripType}
                onChange={setTripType}
                options={[
                  { value: 'roundtrip', label: 'Round-trip' },
                  { value: 'oneway', label: 'One-way' },
                  { value: 'multicity', label: 'Multi-city' },
                ]}
              />
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
              <div>
                <Label htmlFor="origin">From</Label>
                <Input
                  type="text"
                  id="origin"
                  value={origin}
                  onChange={(e) => setOrigin(e.target.value)}
                  placeholder="Airport or city"
                  icon={<PlaneIcon />}
                  required
                />
              </div>

              <div>
                <Label htmlFor="destination">To</Label>
                <Input
                  type="text"
                  id="destination"
                  value={destination}
                  onChange={(e) => setDestination(e.target.value)}
                  placeholder="Airport or city"
                  icon={<LocationIcon />}
                  required
                />
              </div>

              <div>
                <Label htmlFor="departDate">Departure</Label>
                <DateInput
                  id="departDate"
                  value={departDate}
                  onChange={(e) => setDepartDate(e.target.value)}
                  required
                />
              </div>

              <div>
                <Label htmlFor="returnDate">Return</Label>
                {tripType === 'roundtrip' ? (
                  <DateInput
                    id="returnDate"
                    value={returnDate}
                    onChange={(e) => setReturnDate(e.target.value)}
                    required
                  />
                ) : (
                  <DateInput id="returnDate" disabled />
                )}
              </div>
            </div>

            <div className="grid grid-cols-4 sm:grid-cols-3 lg:grid-cols-4 gap-4 mt-4">
              <div className="sm:col-span-1 lg:col-span-1">
                <Label htmlFor="passengers">Passengers</Label>
                <Dropdown label={`${totalPax} ${totalPax === 1 ? 'passenger' : 'passengers'}`}>
                  <DropdownCounter
                    label="Adults"
                    sublabel="12+ years"
                    value={passengers.adults}
                    min={1}
                    onChange={(v) => setPassengers({ ...passengers, adults: v })}
                  />
                  <DropdownCounter
                    label="Children"
                    sublabel="2-11 years"
                    value={passengers.children}
                    onChange={(v) => setPassengers({ ...passengers, children: v })}
                  />
                  <DropdownCounter
                    label="Infants"
                    sublabel="Under 2"
                    value={passengers.infants}
                    onChange={(v) => setPassengers({ ...passengers, infants: v })}
                  />
                </Dropdown>
              </div>
            </div>

            <div className="mt-6">
              <Button type="submit" fullWidth>
                Search Flights
              </Button>
            </div>
          </form>
        </div>

        <div className="mt-6 text-center text-sm">
          <p>Direct flights available · Flexible booking · Compare 500+ airlines worldwide</p>
        </div>
      </div>
    </div>
  );
}
