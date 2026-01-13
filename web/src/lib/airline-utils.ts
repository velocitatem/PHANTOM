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

import { dateToDaysFromToday, dateToIndex, todayIndex } from './date-utils';

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

export { dateToDaysFromToday, dateToIndex, todayIndex };
