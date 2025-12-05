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

const EPOCH = new Date(0);

export const transformProduct = (p: HotelProduct): Hotel => {
  const { id, room_type, date_index, metadata } = p;
  const checkIn = new Date(EPOCH.getTime() + date_index * 86400000);
  const nights = 1;
  const checkOut = new Date(checkIn.getTime() + nights * 86400000);

  return {
    id,
    name: metadata?.name || room_type,
    roomType: room_type,
    checkIn: checkIn.toLocaleDateString('en-US', { month: 'short', day: 'numeric' }),
    checkOut: checkOut.toLocaleDateString('en-US', { month: 'short', day: 'numeric' }),
    dateIndex: date_index,
    amenities: metadata?.amenities || [],
    pricePerNight: metadata?.base_price || 100,
    nights,
  };
};

// convert date string to days from today
export const dateToDaysFromToday = (dateStr: string): number => {
  const target = new Date(dateStr);
  target.setHours(0, 0, 0, 0);
  const today = new Date();
  today.setHours(0, 0, 0, 0);
  return Math.floor((target.getTime() - today.getTime()) / 86400000);
};

// convert date string to date_index (days since epoch)
export const dateToIndex = (dateStr: string): number => {
  const d = new Date(dateStr);
  return Math.floor((d.getTime() - EPOCH.getTime()) / 86400000);
};

// get current date_index
export const todayIndex = (): number => {
  const now = new Date();
  now.setHours(0, 0, 0, 0);
  return Math.floor((now.getTime() - EPOCH.getTime()) / 86400000);
};
