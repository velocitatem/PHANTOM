'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import type { EventName } from '@/lib/events';

const dispatchInteraction = (eventName: EventName, metadata?: Record<string, unknown>) => {
  const e = new CustomEvent('definedInteraction', {
    detail: { eventName, metadata },
  });
  document.dispatchEvent(e);
};

const NavLink = ({ href, children }: { href: string; children: React.ReactNode }) => {
  const path = usePathname();
  const isActive = path === href;

  return (
    <Link
      href={href}
      className={`px-4 py-2 rounded-md transition-colors ${
        isActive
          ? 'bg-[var(--accent-primary)] text-white font-semibold'
          : 'hover:bg-[var(--accent-primary-light)] text-[var(--text-primary)]'
      }`}
    >
      {children}
    </Link>
  );
};

export default function Navigation() {
  return (
    <nav className="bg-[var(--bg-primary)] border-b border-gray-200 shadow-sm">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between h-16">
          <div className="flex items-center space-x-1">
            <NavLink href="/">Home</NavLink>
            <NavLink href="/products">Products</NavLink>
            <NavLink href="/cart">Cart</NavLink>
          </div>
        </div>
      </div>
    </nav>
  );
}
