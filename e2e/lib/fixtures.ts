import { test as base, expect } from '@playwright/test';
import { PhantomApiClient, apiClient } from './api-client';
import { EventGenerator, TestProducts } from './event-generator';
import { runPricingPipeline, waitForPriceUpdate, PipelineResult } from './pipeline-runner';
import { testConfig } from '../playwright.config';

/**
 * Extended test fixtures for PHANTOM E2E tests
 */
export interface PhantomTestFixtures {
  /** API client for interacting with PHANTOM services */
  api: PhantomApiClient;

  /** Event generator for creating test events */
  events: EventGenerator;

  /** Run the pricing pipeline and wait for updates */
  triggerPriceUpdate: (options?: {
    storeMode?: 'hotel' | 'airline';
    highThreshold?: number;
    lowThreshold?: number;
    surgeMultiplier?: number;
    discountMultiplier?: number;
  }) => Promise<PipelineResult>;

  /** Wait for a specific price condition */
  waitForPrice: (
    productId: string,
    condition: (price: number, basePrice: number) => boolean,
    storeMode?: 'hotel' | 'airline'
  ) => Promise<{ price: number; basePrice: number; markup: number }>;

  /** Test configuration */
  config: typeof testConfig;
}

/**
 * Custom test with PHANTOM fixtures
 */
export const test = base.extend<PhantomTestFixtures>({
  api: async ({}, use) => {
    await use(apiClient);
  },

  events: async ({}, use) => {
    // Create a new event generator with a fresh session for each test
    const generator = new EventGenerator({
      storeMode: 'hotel',
    });
    await use(generator);
  },

  triggerPriceUpdate: async ({}, use) => {
    const trigger = async (options = {}) => {
      const result = await runPricingPipeline({
        storeMode: 'hotel',
        highThreshold: testConfig.pricing.highThreshold,
        lowThreshold: testConfig.pricing.lowThreshold,
        surgeMultiplier: testConfig.pricing.surgeMultiplier,
        discountMultiplier: testConfig.pricing.discountMultiplier,
        ...options,
      });

      // Wait a moment for Redis to be fully updated
      await new Promise(resolve => setTimeout(resolve, 500));

      return result;
    };

    await use(trigger);
  },

  waitForPrice: async ({ api }, use) => {
    const waiter = async (
      productId: string,
      condition: (price: number, basePrice: number) => boolean,
      storeMode: 'hotel' | 'airline' = 'hotel'
    ) => {
      let lastPrice = 0;
      let lastBasePrice = 0;

      const updated = await waitForPriceUpdate(async () => {
        const priceResponse = await api.getPrice(storeMode, productId);
        lastPrice = priceResponse.price;
        lastBasePrice = priceResponse.base_price;
        return condition(priceResponse.price, priceResponse.base_price);
      });

      if (!updated) {
        throw new Error(
          `Price condition not met within timeout. Last price: ${lastPrice}, base: ${lastBasePrice}`
        );
      }

      return {
        price: lastPrice,
        basePrice: lastBasePrice,
        markup: lastPrice / lastBasePrice,
      };
    };

    await use(waiter);
  },

  config: async ({}, use) => {
    await use(testConfig);
  },
});

export { expect };

/**
 * Helper assertions for pricing tests
 */
export const PricingAssertions = {
  /**
   * Assert that a price has surge markup applied
   */
  isSurged: (price: number, basePrice: number, expectedMultiplier: number, tolerance = 0.01) => {
    const actualMarkup = price / basePrice;
    const minExpected = expectedMultiplier * (1 - tolerance);
    const maxExpected = expectedMultiplier * (1 + tolerance);
    return actualMarkup >= minExpected && actualMarkup <= maxExpected;
  },

  /**
   * Assert that a price has discount applied
   */
  isDiscounted: (price: number, basePrice: number, expectedMultiplier: number, tolerance = 0.01) => {
    const actualMarkup = price / basePrice;
    const minExpected = expectedMultiplier * (1 - tolerance);
    const maxExpected = expectedMultiplier * (1 + tolerance);
    return actualMarkup >= minExpected && actualMarkup <= maxExpected;
  },

  /**
   * Assert that a price is at base (no surge/discount)
   */
  isBase: (price: number, basePrice: number, tolerance = 0.01) => {
    const actualMarkup = price / basePrice;
    return actualMarkup >= (1 - tolerance) && actualMarkup <= (1 + tolerance);
  },
};
