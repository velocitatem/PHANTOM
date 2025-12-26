import { spawn } from 'child_process';
import path from 'path';
import { testConfig } from '../playwright.config';

/**
 * Pipeline execution result
 */
export interface PipelineResult {
  success: boolean;
  interactions_count: number;
  products_count: number;
  prices_published: boolean;
  prices?: Array<{
    productId: string;
    current_price: number;
    base_price: number;
    optimal_price: number;
    demand_score: number;
  }>;
  timestamp?: string;
  message?: string;
  error?: string;
}

/**
 * Pipeline configuration options
 */
export interface PipelineOptions {
  storeMode?: 'hotel' | 'airline';
  highThreshold?: number;
  lowThreshold?: number;
  surgeMultiplier?: number;
  discountMultiplier?: number;
  dryRun?: boolean;
}

/**
 * Execute the pricing pipeline to update prices based on current events
 */
export async function runPricingPipeline(options: PipelineOptions = {}): Promise<PipelineResult> {
  const {
    storeMode = 'hotel',
    highThreshold = testConfig.pricing.highThreshold,
    lowThreshold = testConfig.pricing.lowThreshold,
    surgeMultiplier = testConfig.pricing.surgeMultiplier,
    discountMultiplier = testConfig.pricing.discountMultiplier,
    dryRun = false,
  } = options;

  const workerPath = path.join(__dirname, 'pipeline-worker.py');

  const args = [
    workerPath,
    '--store-mode', storeMode,
    '--high-threshold', String(highThreshold),
    '--low-threshold', String(lowThreshold),
    '--surge-multiplier', String(surgeMultiplier),
    '--discount-multiplier', String(discountMultiplier),
    '--json-output',
  ];

  if (dryRun) {
    args.push('--dry-run');
  }

  return new Promise((resolve, reject) => {
    const python = spawn('python3', args, {
      env: {
        ...process.env,
        BACKEND_URL: testConfig.backendUrl,
        REDIS_HOST: testConfig.redisHost,
        REDIS_PORT: String(testConfig.redisPort),
        KAFKA_HOST: testConfig.kafkaHost,
        KAFKA_PORT: String(testConfig.kafkaPort),
      },
    });

    let stdout = '';
    let stderr = '';

    python.stdout.on('data', (data) => {
      stdout += data.toString();
    });

    python.stderr.on('data', (data) => {
      stderr += data.toString();
      // Log pipeline output for debugging
      console.log('[Pipeline]', data.toString().trim());
    });

    python.on('close', (code) => {
      if (code === 0) {
        try {
          // Find JSON output in stdout (last JSON object)
          const jsonMatch = stdout.match(/\{[\s\S]*\}$/);
          if (jsonMatch) {
            const result = JSON.parse(jsonMatch[0]);
            resolve(result);
          } else {
            resolve({
              success: true,
              interactions_count: 0,
              products_count: 0,
              prices_published: false,
              message: 'Pipeline completed but no JSON output',
            });
          }
        } catch (parseError) {
          resolve({
            success: true,
            interactions_count: 0,
            products_count: 0,
            prices_published: false,
            message: 'Pipeline completed but output not parseable',
          });
        }
      } else {
        reject(new Error(`Pipeline exited with code ${code}: ${stderr}`));
      }
    });

    python.on('error', (error) => {
      reject(new Error(`Failed to start pipeline: ${error.message}`));
    });
  });
}

/**
 * Wait for prices to be updated in Redis and available via the pricing API
 */
export async function waitForPriceUpdate(
  checkFn: () => Promise<boolean>,
  maxWaitMs: number = testConfig.timing.maxPriceWait,
  intervalMs: number = testConfig.timing.priceCheckInterval
): Promise<boolean> {
  const startTime = Date.now();

  while (Date.now() - startTime < maxWaitMs) {
    try {
      const updated = await checkFn();
      if (updated) {
        return true;
      }
    } catch (error) {
      // Ignore errors during polling
    }

    await new Promise(resolve => setTimeout(resolve, intervalMs));
  }

  return false;
}
