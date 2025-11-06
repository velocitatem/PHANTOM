import { ReactNode } from 'react';
import '@/styles/hotel.css';

export default function HotelLayout({ children }: { children: ReactNode }) {
  return <div data-mode="hotel">{children}</div>;
}
