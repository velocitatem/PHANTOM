import { test, expect } from '../fixtures';
import {
  createFreshSession,
  viewProductViaFlow,
  rapidViewProductViaFlow,
  humanLikeViewProduct,
  getPriceFromDOM,
  verifySessionConsistency,
  addToCart,
} from '../helpers/interactions';
import { getSessionEvents } from '../helpers/kafka';
import { runSessionPricing } from '../helpers/airflow';

test.describe('SessionAwarePricer E2E', () => {
  const STORE_TYPE = 'hotel';

  test('baseline: human-like behavior maintains base price', async ({ page, backendUrl }) => {
    const sessionId = await createFreshSession(page, STORE_TYPE);

    const productId1 = await humanLikeViewProduct(page, STORE_TYPE);
    const baselinePrice = await getPriceFromDOM(page);
    expect(await verifySessionConsistency(page, sessionId)).toBeTruthy();

    await page.waitForTimeout(1500);

    const productId2 = await humanLikeViewProduct(page, STORE_TYPE);

    await runSessionPricing(STORE_TYPE);

    const secondPrice = await getPriceFromDOM(page);
    expect(await verifySessionConsistency(page, sessionId)).toBeTruthy();

    expect(Math.abs(secondPrice - baselinePrice) / baselinePrice).toBeLessThan(0.1);
  });

  test('agent detection: rapid robot-like behavior increases price', async ({ page, backendUrl }) => {
    const sessionId = await createFreshSession(page, STORE_TYPE);

    const productId = await viewProductViaFlow(page, STORE_TYPE);
    const baselinePrice = await getPriceFromDOM(page);

    await page.waitForTimeout(500);

    await rapidViewProductViaFlow(page, 8, 100, STORE_TYPE);
    expect(await verifySessionConsistency(page, sessionId)).toBeTruthy();

    await page.waitForTimeout(1000);

    const events = await getSessionEvents(backendUrl, sessionId);
    expect(events.length).toBeGreaterThanOrEqual(8);

    await runSessionPricing(STORE_TYPE);

    await page.goto(`/products/${productId}`);
    await page.waitForLoadState('networkidle');
    const agentPrice = await getPriceFromDOM(page);

    expect(agentPrice).toBeGreaterThan(baselinePrice);
    expect((agentPrice - baselinePrice) / baselinePrice).toBeGreaterThan(0.01);
  });

  test('velocity threshold: high event rate triggers detection', async ({ page, backendUrl }) => {
    const sessionId = await createFreshSession(page, STORE_TYPE);

    const productId = await viewProductViaFlow(page, STORE_TYPE);
    const baselinePrice = await getPriceFromDOM(page);

    await rapidViewProductViaFlow(page, 10, 80, STORE_TYPE);

    const events = await getSessionEvents(backendUrl, sessionId);
    expect(events.length).toBeGreaterThanOrEqual(10);

    await runSessionPricing(STORE_TYPE);

    await page.goto(`/products/${productId}`);
    await page.waitForLoadState('networkidle');
    const agentPrice = await getPriceFromDOM(page);

    expect(agentPrice).toBeGreaterThan(baselinePrice);
    expect(await verifySessionConsistency(page, sessionId)).toBeTruthy();
  });

  test('cart ratio: high cart/view ratio signals intent', async ({ page, backendUrl }) => {
    const sessionId = await createFreshSession(page, STORE_TYPE);

    const productId = await viewProductViaFlow(page, STORE_TYPE);
    const baselinePrice = await getPriceFromDOM(page);

    await page.waitForTimeout(500);
    await addToCart(page);

    await page.waitForTimeout(2000);

    await page.goto(`/products/${productId}`);
    await page.waitForLoadState('networkidle');
    const cartPrice = await getPriceFromDOM(page);

    expect(cartPrice).toBeGreaterThanOrEqual(baselinePrice);
    expect(await verifySessionConsistency(page, sessionId)).toBeTruthy();
  });

  test('mixed behavior: occasional fast actions tolerated', async ({ page, backendUrl }) => {
    const sessionId = await createFreshSession(page, STORE_TYPE);

    const productId1 = await humanLikeViewProduct(page, STORE_TYPE);
    const baselinePrice = await getPriceFromDOM(page);

    await page.waitForTimeout(1200);

    await rapidViewProductViaFlow(page, 2, 150, STORE_TYPE);

    await page.waitForTimeout(1000);
    await humanLikeViewProduct(page, STORE_TYPE);

    await runSessionPricing(STORE_TYPE);

    const finalPrice = await getPriceFromDOM(page);

    expect(Math.abs(finalPrice - baselinePrice) / baselinePrice).toBeLessThan(0.3);
    expect(await verifySessionConsistency(page, sessionId)).toBeTruthy();
  });

  test('session isolation: agent behavior in one session does not affect others', async ({
    page,
    context,
    backendUrl,
  }) => {
    const sessionIdA = await createFreshSession(page, STORE_TYPE);
    const productId = await viewProductViaFlow(page, STORE_TYPE);
    const basePrice = await getPriceFromDOM(page);

    await rapidViewProductViaFlow(page, 10, 100, STORE_TYPE);
    await page.waitForTimeout(2000);

    await page.goto(`/products/${productId}`);
    await page.waitForLoadState('networkidle');
    const agentPrice = await getPriceFromDOM(page);
    expect(agentPrice).toBeGreaterThan(basePrice * 0.99);

    const page2 = await context.newPage();
    const sessionIdB = await createFreshSession(page2, STORE_TYPE);

    await page2.goto(`/products/${productId}`);
    await page2.waitForLoadState('networkidle');
    const cleanPrice = await getPriceFromDOM(page2);

    expect(Math.abs(cleanPrice - basePrice) / basePrice).toBeLessThan(0.1);
    expect(sessionIdA).not.toBe(sessionIdB);
  });

  test('session persistence: session ID maintained across views', async ({ page }) => {
    const sessionId = await createFreshSession(page, STORE_TYPE);

    await viewProductViaFlow(page, STORE_TYPE);
    expect(await verifySessionConsistency(page, sessionId)).toBeTruthy();

    await viewProductViaFlow(page, STORE_TYPE);
    expect(await verifySessionConsistency(page, sessionId)).toBeTruthy();

    await viewProductViaFlow(page, STORE_TYPE);
    expect(await verifySessionConsistency(page, sessionId)).toBeTruthy();
  });
});
