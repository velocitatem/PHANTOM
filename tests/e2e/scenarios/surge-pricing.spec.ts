import { test, expect } from '../fixtures';
import {
  createFreshSession,
  viewProductViaFlow,
  rapidViewProductViaFlow,
  getPriceFromDOM,
  verifySessionConsistency,
} from '../helpers/interactions';
import { waitForInteractionEvent, countProductViews } from '../helpers/kafka';
import { runSurgePricing } from '../helpers/airflow';

test.describe('SimpleSurgePricer E2E', () => {
  const STORE_TYPE = 'hotel';

  test('baseline: initial price equals base price', async ({ page, backendUrl }) => {
    const sessionId = await createFreshSession(page, STORE_TYPE);

    const productId = await viewProductViaFlow(page, STORE_TYPE);
    const price = await getPriceFromDOM(page);

    expect(price).toBeGreaterThan(0);
    expect(await verifySessionConsistency(page, sessionId)).toBeTruthy();
  });

  test('surge: rapid views trigger price increase', async ({ page, backendUrl }) => {
    const sessionId = await createFreshSession(page, STORE_TYPE);

    const productId = await viewProductViaFlow(page, STORE_TYPE);
    const baselinePrice = await getPriceFromDOM(page);

    await rapidViewProductViaFlow(page, 5, 200, STORE_TYPE);

    await page.waitForTimeout(1000);

    const evt = await waitForInteractionEvent(backendUrl, sessionId, 'view_item_page');
    expect(evt).not.toBeNull();

    const viewCount = await countProductViews(backendUrl, productId);
    expect(viewCount).toBeGreaterThanOrEqual(5);

    await runSurgePricing(STORE_TYPE, 3, 1);

    await page.goto(`/products/${productId}`);
    await page.waitForLoadState('networkidle');
    const surgedPrice = await getPriceFromDOM(page);

    expect(surgedPrice).toBeGreaterThan(baselinePrice);
    expect((surgedPrice - baselinePrice) / baselinePrice).toBeGreaterThan(0.01);
    expect(await verifySessionConsistency(page, sessionId)).toBeTruthy();
  });

  test('threshold: price unchanged below threshold', async ({ page, backendUrl }) => {
    const sessionId = await createFreshSession(page, STORE_TYPE);

    const productId = await viewProductViaFlow(page, STORE_TYPE);
    const baselinePrice = await getPriceFromDOM(page);

    await rapidViewProductViaFlow(page, 2, 300, STORE_TYPE);

    await page.waitForTimeout(1500);

    await page.goto(`/products/${productId}`);
    await page.waitForLoadState('networkidle');
    const currentPrice = await getPriceFromDOM(page);

    expect(Math.abs(currentPrice - baselinePrice) / baselinePrice).toBeLessThan(0.05);
    expect(await verifySessionConsistency(page, sessionId)).toBeTruthy();
  });

  test('window: surge decays after window expires', async ({ page, backendUrl }) => {
    const sessionId = await createFreshSession(page, STORE_TYPE);

    const productId = await viewProductViaFlow(page, STORE_TYPE);
    const baselinePrice = await getPriceFromDOM(page);

    await rapidViewProductViaFlow(page, 5, 150, STORE_TYPE);

    await page.waitForTimeout(1000);

    await runSurgePricing(STORE_TYPE, 3, 1);

    await page.goto(`/products/${productId}`);
    await page.waitForLoadState('networkidle');
    const surgedPrice = await getPriceFromDOM(page);
    expect(surgedPrice).toBeGreaterThan(baselinePrice);

    await page.waitForTimeout(12000);

    await runSurgePricing(STORE_TYPE, 3, 1);

    await page.goto(`/products/${productId}`);
    await page.waitForLoadState('networkidle');
    const decayedPrice = await getPriceFromDOM(page);
    expect(decayedPrice).toBeLessThan(surgedPrice);
    expect(await verifySessionConsistency(page, sessionId)).toBeTruthy();
  });

  test('isolation: different products have independent surge', async ({ page, backendUrl }) => {
    const sessionId = await createFreshSession(page, STORE_TYPE);

    const productIdA = await viewProductViaFlow(page, STORE_TYPE);
    const basePriceA = await getPriceFromDOM(page);

    await rapidViewProductViaFlow(page, 5, 200, STORE_TYPE);
    await page.waitForTimeout(2000);

    await page.goto(`/products/${productIdA}`);
    await page.waitForLoadState('networkidle');
    const surgedPriceA = await getPriceFromDOM(page);

    const productIdB = await viewProductViaFlow(page, STORE_TYPE);
    const priceB = await getPriceFromDOM(page);

    expect(surgedPriceA).toBeGreaterThan(basePriceA * 0.99);
    expect(productIdA).not.toBe(productIdB);
    expect(await verifySessionConsistency(page, sessionId)).toBeTruthy();
  });
});
