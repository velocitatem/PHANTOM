export interface AirlineProduct {
  id: string;
  flight_type: string;
  date_index: number;
  metadata: {
    departure: { time: string; airport: string };
    arrival: { time: string; airport: string };
    duration: string;
    stops: number;
    cabin_class: string;
    fare_rule: string;
    refundable: boolean;
    total?: number;
    base_price: number;
  };
  availability: number;
}

export interface Flight {
  id: string;
  flightType: string;
  departure: { time: string; airport: string };
  arrival: { time: string; airport: string };
  duration: string;
  stops: number;
  cabinClass: string;
  fareRule: string;
  refundable: boolean;
  basePrice: number;
  dateIndex: number;
  availability: number;
}

const EPOCH = new Date(0);

export const transformProduct = (p: AirlineProduct): Flight => {
  const { id, flight_type, date_index, metadata, availability } = p;

  return {
    id,
    flightType: flight_type,
    departure: metadata.departure,
    arrival: metadata.arrival,
    duration: metadata.duration,
    stops: metadata.stops,
    cabinClass: metadata.cabin_class,
    fareRule: metadata.fare_rule,
    refundable: metadata.refundable,
    basePrice: metadata.base_price,
    dateIndex: date_index,
    availability,
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
