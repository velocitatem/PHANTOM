import { HotelProduct, Hotel, transformProduct as transformHotel } from './hotel-utils';
import { AirlineProduct, Flight, transformProduct as transformFlight } from './airline-utils';

export type Product = Hotel | Flight;
export type ProductRaw = HotelProduct | AirlineProduct;

export const isHotelProduct = (p: ProductRaw): p is HotelProduct => {
  return 'room_type' in p;
};

export const isAirlineProduct = (p: ProductRaw): p is AirlineProduct => {
  return 'flight_type' in p;
};

export const transformProduct = (p: ProductRaw): Product => {
  if (isHotelProduct(p)) {
    return transformHotel(p);
  }
  return transformFlight(p);
};

export const getProductType = (p: Product): 'hotel' | 'airline' => {
  if ('roomType' in p) return 'hotel';
  return 'airline';
};
