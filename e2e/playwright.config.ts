import { defineConfig, devices } from '@playwright/test';

/**
 * Playwright configuration for PHANTOM Dynamic Pricing E2E Tests
 *
 * Tests validate the entire pricing pipeline:
 * Frontend Events → Kafka → Pipeline Processing → Redis → Pricing API
 */
export default defineConfig({
  testDir: './tests',
  fullyParallel: false, // Run tests sequentially to avoid race conditions in shared state
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 2 : 0,
  workers: 1, // Single worker for E2E tests to ensure isolation
  reporter: [
    ['html', { outputFolder: 'playwright-report' }],
    ['list']
  ],

  // Global timeout for each test
  timeout: 120_000, // 2 minutes per test (includes pipeline processing time)

  // Expect timeout for assertions
  expect: {
    timeout: 30_000, // 30 seconds for price updates to propagate
  },

  use: {
    // Base URL for the backend API
    baseURL: process.env.BACKEND_URL || 'http://localhost:5000',

    // Collect trace on first retry
    trace: 'on-first-retry',

    // Screenshot on failure
    screenshot: 'only-on-failure',
  },

  // Global setup and teardown
  globalSetup: require.resolve('./global-setup'),
  globalTeardown: require.resolve('./global-teardown'),

  projects: [
    {
      name: 'dynamic-pricing',
      testMatch: /.*\.spec\.ts/,
    },
  ],

  // Environment configuration
  // These can be overridden via environment variables
});

// Export test configuration constants
export const testConfig = {
  // API endpoints
  backendUrl: process.env.BACKEND_URL || 'http://localhost:5000',
  providerUrl: process.env.PROVIDER_URL || 'http://localhost:5001',

  // Redis configuration
  redisHost: process.env.REDIS_HOST || 'localhost',
  redisPort: parseInt(process.env.REDIS_PORT || '6378'),

  // Kafka configuration
  kafkaHost: process.env.KAFKA_HOST || 'localhost',
  kafkaPort: parseInt(process.env.KAFKA_PORT || '9092'),

  // Pricing thresholds for tests (aggressive settings for fast feedback)
  pricing: {
    highThreshold: 3,     // Trigger surge after 3 interactions
    lowThreshold: 1,      // Trigger discount at 1 or fewer interactions
    surgeMultiplier: 1.5, // 50% price increase on surge
    discountMultiplier: 0.9, // 10% discount on low demand
    windowSize: 10_000,   // 10 second window for demand calculation
  },

  // Timing configuration
  timing: {
    eventDelay: 100,           // Delay between events (ms)
    pipelineWait: 5_000,       // Wait for pipeline processing (ms)
    priceCheckInterval: 1_000, // Interval between price checks (ms)
    maxPriceWait: 30_000,      // Max wait for price update (ms)
  },
};
