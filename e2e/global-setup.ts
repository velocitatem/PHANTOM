import { testConfig } from './playwright.config';

/**
 * Global setup for E2E tests
 * Verifies all services are healthy before running tests
 */
async function globalSetup() {
  console.log('\n🚀 PHANTOM E2E Test Suite - Global Setup\n');

  // Check backend health
  await checkService('Backend API', `${testConfig.backendUrl}/health`);

  // Check pricing provider health
  await checkService('Pricing Provider', `${testConfig.providerUrl}/health`);

  console.log('\n✅ All services healthy. Starting tests...\n');
}

async function checkService(name: string, url: string): Promise<void> {
  const maxRetries = 10;
  const retryDelay = 2000;

  for (let attempt = 1; attempt <= maxRetries; attempt++) {
    try {
      const response = await fetch(url);
      if (response.ok) {
        const data = await response.json();
        console.log(`✅ ${name}: healthy`);
        if (data.redis !== undefined) {
          console.log(`   └─ Redis: ${data.redis ? 'connected' : 'disconnected'}`);
        }
        if (data.kafka !== undefined) {
          console.log(`   └─ Kafka: ${data.kafka}`);
        }
        return;
      }
    } catch (error) {
      if (attempt === maxRetries) {
        throw new Error(`❌ ${name} is not available at ${url} after ${maxRetries} attempts`);
      }
      console.log(`⏳ Waiting for ${name} (attempt ${attempt}/${maxRetries})...`);
      await new Promise(resolve => setTimeout(resolve, retryDelay));
    }
  }
}

export default globalSetup;
