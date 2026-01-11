import { test, expect } from '../fixtures';
import { createFreshSession, navigateToProduct, rapidViewProduct } from '../helpers/interactions';
import { fetchPrice, waitForPriceChange } from '../helpers/api';
import { waitForInteractionEvent, countProductViews } from '../helpers/kafka';

test.describe('SimpleSurgePricer E2E', () => {
  const PRODUCT_ID = 'hotel_001';
  const PRICING_MODE = 'simple_surge';

  test('baseline: initial price equals base price', async ({ page, backendUrl }) => {
    await createFreshSession(page);
    await navigateToProduct(page, PRODUCT_ID);

    const priceResp = await fetchPrice(page.url(), PRODUCT_ID, PRICING_MODE);

    expect(priceResp.price).toBeCloseTo(priceResp.base_price, 2);
    expect(priceResp.markup).toBeCloseTo(0, 2);
  });

  test('surge: rapid views trigger price increase', async ({ page, backendUrl }) => {
    const sessionId = await createFreshSession(page);

    const baselineResp = await fetchPrice(page.url(), PRODUCT_ID, PRICING_MODE, sessionId);
    const baselinePrice = baselineResp.price;

    await rapidViewProduct(page, PRODUCT_ID, 5, 200);

    await page.waitForTimeout(2000);

    const evt = await waitForInteractionEvent(backendUrl, sessionId, 'view_item_page');
    expect(evt).not.toBeNull();

    const viewCount = await countProductViews(backendUrl, PRODUCT_ID);
    expect(viewCount).toBeGreaterThanOrEqual(5);

    const surgedResp = await waitForPriceChange(
      page.url(),
      PRODUCT_ID,
      baselinePrice,
      PRICING_MODE,
      sessionId
    );

    expect(surgedResp.price).toBeGreaterThan(baselinePrice);
    expect(surgedResp.markup).toBeGreaterThan(0);

    const expectedSurge = baselineResp.base_price * 1.5;
    expect(surgedResp.price).toBeCloseTo(expectedSurge, 1);
  });

  test('threshold: price unchanged below threshold', async ({ page, backendUrl }) => {
    const sessionId = await createFreshSession(page);

    const baselineResp = await fetchPrice(page.url(), PRODUCT_ID, PRICING_MODE, sessionId);
    const baselinePrice = baselineResp.price;

    await rapidViewProduct(page, PRODUCT_ID, 2, 300);

    await page.waitForTimeout(1500);

    const currentResp = await fetchPrice(page.url(), PRODUCT_ID, PRICING_MODE, sessionId);

    expect(currentResp.price).toBeCloseTo(baselinePrice, 2);
    expect(currentResp.markup).toBeCloseTo(0, 2);
  });

  test('window: surge decays after window expires', async ({ page, backendUrl }) => {
    const sessionId = await createFreshSession(page);

    await rapidViewProduct(page, PRODUCT_ID, 5, 150);

    await page.waitForTimeout(1500);

    const surgedResp = await fetchPrice(page.url(), PRODUCT_ID, PRICING_MODE, sessionId);
    expect(surgedResp.price).toBeGreaterThan(surgedResp.base_price);

    await page.waitForTimeout(12000);

    const decayedResp = await fetchPrice(page.url(), PRODUCT_ID, PRICING_MODE, sessionId);
    expect(decayedResp.price).toBeLessThan(surgedResp.price);
  });

  test('isolation: different products have independent surge', async ({ page, backendUrl }) => {
    const sessionId = await createFreshSession(page);
    const PRODUCT_A = 'hotel_001';
    const PRODUCT_B = 'hotel_002';

    await rapidViewProduct(page, PRODUCT_A, 5, 200);

    await page.waitForTimeout(2000);

    const priceA = await fetchPrice(page.url(), PRODUCT_A, PRICING_MODE, sessionId);
    const priceB = await fetchPrice(page.url(), PRODUCT_B, PRICING_MODE, sessionId);

    expect(priceA.price).toBeGreaterThan(priceA.base_price);
    expect(priceB.price).toBeCloseTo(priceB.base_price, 2);
  });
});
