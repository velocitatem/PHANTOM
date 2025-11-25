'use client';

import { createContext, useContext, useState, useEffect, ReactNode } from 'react';

export interface CartItem {
  id: string;
  type: 'hotel' | 'airline';
  name: string;
  price: number;
  metadata: Record<string, unknown>;
  dateIndex: number;
}

interface CartContextType {
  items: CartItem[];
  addItem: (item: CartItem) => void;
  removeItem: (id: string) => void;
  clearCart: () => void;
  itemCount: number;
}

const CartContext = createContext<CartContextType | undefined>(undefined);

const CART_KEY = 'phantom_cart';

export const CartProvider = ({ children }: { children: ReactNode }) => {
  const [items, setItems] = useState<CartItem[]>([]);
  const [loaded, setLoaded] = useState(false);

  // load cart from sessionStorage on mount
  useEffect(() => {
    const stored = sessionStorage.getItem(CART_KEY);
    if (stored) {
      try {
        setItems(JSON.parse(stored));
      } catch (e) {
        console.error('[CART_LOAD]', e);
      }
    }
    setLoaded(true);
  }, []);

  // persist to sessionStorage whenever cart changes
  useEffect(() => {
    if (!loaded) return;
    sessionStorage.setItem(CART_KEY, JSON.stringify(items));
  }, [items, loaded]);

  const addItem = (item: CartItem) => {
    setItems(prev => {
      // prevent duplicates
      if (prev.find(i => i.id === item.id)) return prev;
      return [...prev, item];
    });
  };

  const removeItem = (id: string) => {
    setItems(prev => prev.filter(i => i.id !== id));
  };

  const clearCart = () => {
    setItems([]);
  };

  return (
    <CartContext.Provider value={{ items, addItem, removeItem, clearCart, itemCount: items.length }}>
      {children}
    </CartContext.Provider>
  );
};

export const useCart = () => {
  const ctx = useContext(CartContext);
  if (!ctx) throw new Error('useCart must be used within CartProvider');
  return ctx;
};
