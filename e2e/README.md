# PHANTOM Dynamic Pricing E2E Test Suite

End-to-end tests validating the dynamic pricing pipeline, including SimpleSurgePricer and SessionAwarePricer functionality.

## System Under Test (SUT)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        PHANTOM Pricing Pipeline                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────────┐  │
│  │  Test Runner │───▶│ Backend API  │───▶│ Kafka (user-interactions)│  │
│  │  (Playwright)│    │ POST /ingest │    │                          │  │
│  └──────────────┘    └──────────────┘    └────────────┬─────────────┘  │
│         │                                              │                │
│         │                                              ▼                │
│         │                                 ┌──────────────────────────┐  │
│         │                                 │   Pipeline Worker        │  │
│         │                                 │   - Fetch interactions   │  │
│         │                                 │   - Compute demand       │  │
│         │                                 │   - Apply surge pricing  │  │
│         │                                 └────────────┬─────────────┘  │
│         │                                              │                │
│         │                                              ▼                │
│         │                                 ┌──────────────────────────┐  │
│         │                                 │   Redis (Model Registry) │  │
│         │                                 │   - prices:latest        │  │
│         │                                 └────────────┬─────────────┘  │
│         │                                              │                │
│         │                                              ▼                │
│         │     ┌──────────────┐           ┌──────────────────────────┐  │
│         └────▶│ Pricing API  │◀──────────│   Pricing Provider       │  │
│               │ GET /price   │           │   (serves from Redis)    │  │
│               └──────────────┘           └──────────────────────────┘  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## Test Scenarios

| Scenario | Description | Expected Outcome |
|----------|-------------|------------------|
| **Baseline** | No interactions for product | Price = base_price (markup = 1.0) |
| **Surge** | 5+ interactions (above threshold) | Price = base_price × 1.5 |
| **Discount** | 1 interaction (at threshold) | Price = base_price × 0.9 |
| **Multi-Product** | Different demand per product | Each product priced by its demand |
| **Propagation** | Pipeline → Redis → API | Prices visible via API |
| **Event Types** | Mix of view, click, cart | All events counted in demand |
| **Multi-Session** | Events from different sessions | Demand aggregated correctly |

## Test Configuration

The tests use aggressive thresholds for fast feedback:

```typescript
pricing: {
  highThreshold: 3,      // Surge after 3 interactions
  lowThreshold: 1,       // Discount at ≤1 interaction
  surgeMultiplier: 1.5,  // 50% price increase
  discountMultiplier: 0.9, // 10% discount
  windowSize: 10_000,    // 10 second window
}
```

## Quick Start

### 1. Start E2E Services

```bash
# Start minimal services for E2E testing
docker compose -f docker-compose.e2e.yml up -d

# Wait for services to be healthy
docker compose -f docker-compose.e2e.yml ps

# Optional: Start with Kafka UI for debugging
docker compose -f docker-compose.e2e.yml --profile debug up -d
```

### 2. Install Test Dependencies

```bash
cd e2e
npm install
npx playwright install
```

### 3. Run Tests

```bash
# Run all E2E tests
npm test

# Run with UI (interactive mode)
npm run test:ui

# Run specific test file
npm run test:pricing

# Run in debug mode
npm run test:debug

# View test report
npm run test:report
```

### 4. Cleanup

```bash
docker compose -f docker-compose.e2e.yml down -v
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `BACKEND_URL` | `http://localhost:5000` | Backend API URL |
| `PROVIDER_URL` | `http://localhost:5001` | Pricing Provider URL |
| `REDIS_HOST` | `localhost` | Redis host |
| `REDIS_PORT` | `6378` | Redis port |
| `KAFKA_HOST` | `localhost` | Kafka host |
| `KAFKA_PORT` | `9092` | Kafka port |

## Test Architecture

```
e2e/
├── playwright.config.ts    # Playwright configuration
├── global-setup.ts         # Service health checks
├── global-teardown.ts      # Cleanup
├── package.json            # Dependencies and scripts
├── tsconfig.json           # TypeScript configuration
├── lib/
│   ├── api-client.ts       # API interaction utilities
│   ├── event-generator.ts  # Test event factory
│   ├── pipeline-runner.ts  # TypeScript pipeline wrapper
│   ├── pipeline-worker.py  # Python pipeline executor
│   ├── fixtures.ts         # Playwright test fixtures
│   └── index.ts            # Re-exports
└── tests/
    └── dynamic-pricing.spec.ts  # Main test file
```

## Pipeline Worker

The tests use a dedicated Python pipeline worker (`lib/pipeline-worker.py`) instead of Airflow for faster, more reliable test execution.

```bash
# Run pipeline manually
python3 lib/pipeline-worker.py \
  --store-mode hotel \
  --high-threshold 3 \
  --surge-multiplier 1.5 \
  --json-output

# Dry run (no Redis publish)
python3 lib/pipeline-worker.py --dry-run
```

## Debugging

### View Kafka Events

```bash
# Via API
curl "http://localhost:5000/api/kafka/dump?topic=user-interactions&last_n=10"

# Via Redpanda Console (if started with --profile debug)
open http://localhost:8080
```

### Check Redis State

```bash
docker exec -it PHANTOM-e2e-redis redis-cli
> GET prices:latest
> KEYS *
```

### View Pipeline Logs

The pipeline worker logs detailed information:

```
[INFO] Starting E2E pricing pipeline: mode=hotel, high_threshold=3, surge_multiplier=1.5
[INFO] Fetched 15 interaction records
[INFO] Computed demand for 3 products
[INFO] Applied surge pricing:
  e2e-test...: base=$100.00 -> optimal=$150.00 (demand=5, markup=1.50x)
[INFO] Published 3 prices to Redis
```

## Writing New Tests

```typescript
import { test, expect } from '../lib/fixtures';
import { generateTestProductId } from '../lib/event-generator';

test('my new pricing test', async ({ api, events, triggerPriceUpdate }) => {
  // 1. Create unique product ID
  const productId = generateTestProductId('my-test');

  // 2. Log base price
  await api.logPrice({
    productId,
    price: 100.0,
    sessionId: events.session,
    storeMode: 'hotel',
  });

  // 3. Generate events
  const surgeEvents = events.generateSurgeSequence(productId, 5);
  await api.ingestEvents(surgeEvents);

  // 4. Trigger pipeline
  const result = await triggerPriceUpdate();

  // 5. Verify results
  expect(result.success).toBe(true);
  const pricedProduct = result.prices?.find(p => p.productId === productId);
  expect(pricedProduct?.optimal_price).toBeGreaterThan(100);
});
```

## Troubleshooting

### "Backend not available"

Ensure services are running:
```bash
docker compose -f docker-compose.e2e.yml ps
docker compose -f docker-compose.e2e.yml logs backend
```

### "No interactions found"

Check Kafka topic has events:
```bash
curl "http://localhost:5000/api/kafka/dump?topic=user-interactions"
```

### "Pipeline timeout"

Increase timeout in `playwright.config.ts`:
```typescript
timeout: 180_000, // 3 minutes
```

### "Price not updated"

Check Redis has latest prices:
```bash
docker exec -it PHANTOM-e2e-redis redis-cli GET prices:latest
```
