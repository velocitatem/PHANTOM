import { test as base } from '@playwright/test';

type TestFixtures = {
  backendUrl: string;
  pricingUrl: string;
};

export const test = base.extend<TestFixtures>({
  backendUrl: async ({}, use) => {
    await use(process.env.BACKEND_URL || 'http://localhost:5000');
  },
  pricingUrl: async ({}, use) => {
    await use(process.env.PRICING_PROVIDER_URL || 'http://localhost:5001');
  },
});

export { expect } from '@playwright/test';
