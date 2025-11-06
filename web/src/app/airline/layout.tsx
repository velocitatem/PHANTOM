import { ReactNode } from 'react';
import '@/styles/airline.css';

export default function AirlineLayout({ children }: { children: ReactNode }) {
  return <div data-mode="airline">{children}</div>;
}
