import { test, expect } from '../fixtures';
import {
  createFreshSession,
  navigateToProduct,
  rapidViewProduct,
  humanLikeViewProduct,
  addToCart,
} from '../helpers/interactions';
import { fetchPrice, waitForPriceChange } from '../helpers/api';
import { getSessionEvents } from '../helpers/kafka';

test.describe('SessionAwarePricer E2E', () => {
  const PRODUCT_ID = 'hotel_001';
  const PRICING_MODE = 'session_aware';

  test('baseline: human-like behavior maintains base price', async ({ page, backendUrl }) => {
    const sessionId = await createFreshSession(page);

    const baselineResp = await fetchPrice(page.url(), PRODUCT_ID, PRICING_MODE, sessionId);

    await humanLikeViewProduct(page, PRODUCT_ID);
    await page.waitForTimeout(1500);
    await humanLikeViewProduct(page, PRODUCT_ID);

    await page.waitForTimeout(2000);

    const currentResp = await fetchPrice(page.url(), PRODUCT_ID, PRICING_MODE, sessionId);

    expect(currentResp.price).toBeCloseTo(baselineResp.price, 2);
    expect(currentResp.markup).toBeCloseTo(0, 2);
  });

  test('agent detection: rapid robot-like behavior increases price', async ({ page, backendUrl }) => {
    const sessionId = await createFreshSession(page);

    const baselineResp = await fetchPrice(page.url(), PRODUCT_ID, PRICING_MODE, sessionId);
    const baselinePrice = baselineResp.price;

    await rapidViewProduct(page, PRODUCT_ID, 8, 100);

    await page.waitForTimeout(2500);

    const events = await getSessionEvents(backendUrl, sessionId);
    expect(events.length).toBeGreaterThanOrEqual(8);

    const agentResp = await waitForPriceChange(
      page.url(),
      PRODUCT_ID,
      baselinePrice,
      PRICING_MODE,
      sessionId,
      15,
      600
    );

    expect(agentResp.price).toBeGreaterThan(baselinePrice);
    expect(agentResp.markup).toBeGreaterThan(0);
  });

  test('velocity threshold: high event rate triggers detection', async ({ page, backendUrl }) => {
    const sessionId = await createFreshSession(page);

    const baselineResp = await fetchPrice(page.url(), PRODUCT_ID, PRICING_MODE, sessionId);

    const startTime = Date.now();
    await rapidViewProduct(page, PRODUCT_ID, 10, 80);
    const duration = (Date.now() - startTime) / 1000;

    const eventsPerSec = 10 / duration;
    expect(eventsPerSec).toBeGreaterThan(2.0);

    await page.waitForTimeout(2000);

    const agentResp = await fetchPrice(page.url(), PRODUCT_ID, PRICING_MODE, sessionId);
    expect(agentResp.price).toBeGreaterThan(baselineResp.price);
  });

  test('cart ratio: high cart/view ratio signals intent', async ({ page, backendUrl }) => {
    const sessionId = await createFreshSession(page);

    const baselineResp = await fetchPrice(page.url(), PRODUCT_ID, PRICING_MODE, sessionId);

    await navigateToProduct(page, PRODUCT_ID);
    await page.waitForTimeout(500);
    await addToCart(page);

    await page.waitForTimeout(2000);

    const cartResp = await fetchPrice(page.url(), PRODUCT_ID, PRICING_MODE, sessionId);

    expect(cartResp.price).toBeGreaterThanOrEqual(baselineResp.price);
  });

  test('mixed behavior: occasional fast actions tolerated', async ({ page, backendUrl }) => {
    const sessionId = await createFreshSession(page);

    const baselineResp = await fetchPrice(page.url(), PRODUCT_ID, PRICING_MODE, sessionId);

    await humanLikeViewProduct(page, PRODUCT_ID);
    await page.waitForTimeout(1200);

    await rapidViewProduct(page, PRODUCT_ID, 2, 150);

    await page.waitForTimeout(1500);
    await humanLikeViewProduct(page, PRODUCT_ID);

    await page.waitForTimeout(2000);

    const currentResp = await fetchPrice(page.url(), PRODUCT_ID, PRICING_MODE, sessionId);

    expect(Math.abs(currentResp.price - baselineResp.price)).toBeLessThan(
      baselineResp.base_price * 0.2
    );
  });

  test('session isolation: agent behavior in one session does not affect others', async ({
    page,
    context,
    backendUrl,
  }) => {
    const sessionIdA = await createFreshSession(page);
    await rapidViewProduct(page, PRODUCT_ID, 10, 100);
    await page.waitForTimeout(2000);

    const agentResp = await fetchPrice(page.url(), PRODUCT_ID, PRICING_MODE, sessionIdA);
    expect(agentResp.price).toBeGreaterThan(agentResp.base_price);

    const page2 = await context.newPage();
    const sessionIdB = await createFreshSession(page2);

    const cleanResp = await fetchPrice(page2.url(), PRODUCT_ID, PRICING_MODE, sessionIdB);

    expect(cleanResp.price).toBeCloseTo(cleanResp.base_price, 2);
  });
});
