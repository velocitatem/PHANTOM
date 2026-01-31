export interface HotelProduct {
  id: string;
  room_type: string;
  date_index: number;
  metadata: {
    amenities?: string[];
    total?: number;
    image_url?: string;
    base_price?: number;
    name?: string;
    refundable?: boolean;
  };
  availability: number;
}

export interface Hotel {
  id: string;
  name: string;
  roomType: string;
  checkIn: string;
  checkOut: string;
  dateIndex: number;
  amenities: string[];
  pricePerNight: number;
  nights: number;
}

import { EPOCH, MS_PER_DAY, dateToDaysFromToday, dateToIndex, todayIndex } from './date-utils';

export const transformProduct = (p: HotelProduct): Hotel => {
  const { id, room_type, date_index, metadata } = p;

  // DB stores date_index as days since epoch
  // but if value is small (<1000), treat as days from today for backward compat
  let checkIn: Date;
  if (date_index < 1000) {
    // legacy: treat as offset from today
    const today = new Date();
    today.setHours(0, 0, 0, 0);
    checkIn = new Date(today.getTime() + date_index * MS_PER_DAY);
  } else {
    // proper: days since epoch
    checkIn = new Date(EPOCH.getTime() + date_index * MS_PER_DAY);
  }

  const nights = 1;
  const checkOut = new Date(checkIn.getTime() + nights * MS_PER_DAY);

  const formatOpts: Intl.DateTimeFormatOptions = {
    month: 'short',
    day: 'numeric',
    year: checkIn.getFullYear() !== new Date().getFullYear() ? 'numeric' : undefined
  };

  return {
    id,
    name: metadata?.name || room_type,
    roomType: room_type,
    checkIn: checkIn.toLocaleDateString('en-US', formatOpts),
    checkOut: checkOut.toLocaleDateString('en-US', formatOpts),
    dateIndex: date_index,
    amenities: metadata?.amenities || [],
    pricePerNight: metadata?.base_price || 100,
    nights,
  };
};

const hotelImagePool = [
  'photo-1566073771259-6a8506099945',
  'photo-1551882547-ff40c63fe5fa',
  'photo-1590490360182-c33d57733427',
  'photo-1582719478250-c89cae4dc85b',
  'photo-1596701062351-8c2c14d1fdd0',
  'photo-1631049307264-da0ec9d70304',
  'photo-1578683010236-d716f9a3f461',
  'photo-1540518614846-7eded433c457',
  'photo-1505693416388-ac5ce068fe85',
  'photo-1522771739844-6a9f6d5f14af',
  'photo-1562438668-bcf0ca6578f0',
  'photo-1595576508898-0ad5c879a061',
];

const hashString = (s: string): number => {
  let h = 0;
  for (let i = 0; i < s.length; i++) {
    h = ((h << 5) - h) + s.charCodeAt(i);
    h = h & h;
  }
  return Math.abs(h);
};

export const getHotelImageUrl = (hotelId: string, size: { w: number; h: number } = { w: 400, h: 300 }): string => {
  const idx = hashString(hotelId) % hotelImagePool.length;
  const photoId = hotelImagePool[idx];
  return `https://images.unsplash.com/${photoId}?w=${size.w}&h=${size.h}&fit=crop`;
};

export { dateToDaysFromToday, dateToIndex, todayIndex };
