'use client';

import { useInteractionTracking } from '@/hooks/useInteractionTracking';

export const TrackingProvider = ({ children }: { children: React.ReactNode }) => {
  useInteractionTracking();
  return <>{children}</>;
};
