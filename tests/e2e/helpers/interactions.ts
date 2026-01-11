import { Page } from '@playwright/test';

export async function getSessionId(page: Page): Promise<string | null> {
  const cookies = await page.context().cookies();
  const sessionCookie = cookies.find(c => c.name === 'phantom_session_id');
  return sessionCookie?.value || null;
}

export async function verifySessionConsistency(page: Page, expectedSessionId: string): Promise<boolean> {
  const currentSessionId = await getSessionId(page);
  return currentSessionId === expectedSessionId;
}

export async function createFreshSession(page: Page, storeType: 'hotel' | 'airline' = 'hotel'): Promise<string> {
  await page.context().clearCookies();
  await page.goto('/');
  await page.waitForLoadState('networkidle');
  await page.waitForTimeout(500);

  const sid = await getSessionId(page);
  if (!sid) throw new Error('Session not created');
  return sid;
}

interface SearchParams {
  destination?: string;
  checkIn?: string;
  guests?: number;
  rooms?: number;
  origin?: string;
  departure?: string;
  adults?: number;
}

export async function performSearch(
  page: Page,
  params: SearchParams,
  storeType: 'hotel' | 'airline' = 'hotel'
): Promise<void> {
  await page.waitForLoadState('networkidle');

  if (storeType === 'hotel') {
    const destInput = page.locator('input#destination');
    await destInput.fill(params.destination || 'New York');

    const checkInInput = page.locator('input#checkIn');
    const checkInDate = params.checkIn || new Date(Date.now() + 7 * 86400000).toISOString().split('T')[0];
    await checkInInput.fill(checkInDate);

    const searchBtn = page.locator('button:has-text("Search Rooms")');
    await searchBtn.click();
  } else {
    const originDropdown = page.locator('button:has-text("Select origin")').or(
      page.locator('[id="origin"]').locator('button').first()
    );
    await originDropdown.click();
    await page.waitForTimeout(200);
    const originOption = page.locator(`button:has-text("${params.origin || 'JFK'}")`).first();
    await originOption.click();
    await page.waitForTimeout(200);

    const destDropdown = page.locator('button:has-text("Select destination")').or(
      page.locator('[id="destination"]').locator('button').first()
    );
    await destDropdown.click();
    await page.waitForTimeout(200);
    const destOption = page.locator(`button:has-text("${params.destination || 'LAX'}")`).first();
    await destOption.click();
    await page.waitForTimeout(200);

    const departInput = page.locator('input#departDate');
    const departDate = params.departure || new Date(Date.now() + 7 * 86400000).toISOString().split('T')[0];
    await departInput.fill(departDate);

    const searchBtn = page.locator('button:has-text("Search Flights")');
    await searchBtn.click();
  }

  await page.waitForLoadState('networkidle');
}

export async function selectRandomProduct(page: Page, storeType: 'hotel' | 'airline' = 'hotel'): Promise<string> {
  await page.waitForLoadState('networkidle');

  const cardClass = storeType === 'hotel' ? '.hotel-card' : '.flight-card';
  const productCards = page.locator(cardClass);

  const count = await productCards.count();
  if (count === 0) throw new Error('No products found on listing page');

  const randomIdx = Math.floor(Math.random() * count);
  return randomIdx.toString();
}

export async function openProductFromListing(page: Page, productId?: string): Promise<string> {
  await page.waitForLoadState('networkidle');

  const hotelCards = page.locator('.hotel-card');
  const flightCards = page.locator('.flight-card');

  const hotelCount = await hotelCards.count();
  const flightCount = await flightCards.count();

  let productCards;
  if (hotelCount > 0) {
    productCards = hotelCards;
  } else if (flightCount > 0) {
    productCards = flightCards;
  } else {
    throw new Error('No products found on listing page');
  }

  const count = await productCards.count();
  const randomIdx = productId ? 0 : Math.floor(Math.random() * count);
  await productCards.nth(randomIdx).click();

  await page.waitForLoadState('networkidle');

  const url = page.url();
  const match = url.match(/\/products\/([^/?]+)/);
  if (!match) throw new Error('Cannot parse product ID from URL after navigation');

  return match[1];
}

export async function getPriceFromDOM(page: Page): Promise<number> {
  await page.waitForLoadState('networkidle');

  await page.waitForSelector('.price-amount', { timeout: 15000 }).catch(() => null);

  const priceSelectors = [
    '.price-amount',
    '.price-display',
    '[data-testid="price"]',
    '[data-price]',
  ];

  for (const selector of priceSelectors) {
    const priceEl = page.locator(selector).first();
    if (await priceEl.count() > 0) {
      const text = await priceEl.textContent();
      if (!text) continue;

      const match = text.match(/[\$]?\s*([\d,]+(?:\.\d{2})?)/);
      if (match) {
        const priceStr = match[1].replace(/,/g, '');
        return parseFloat(priceStr);
      }
    }
  }

  const dataPrice = await page.locator('[data-price]').first().getAttribute('data-price').catch(() => null);
  if (dataPrice) return parseFloat(dataPrice);

  throw new Error('Cannot extract price from DOM');
}

export async function navigateToProduct(
  page: Page,
  productId: string,
  storeType: 'hotel' | 'airline' = 'hotel'
): Promise<void> {
  await page.goto(`/products/${productId}`);
  await page.waitForLoadState('networkidle');
}

export async function viewProductViaFlow(
  page: Page,
  storeType: 'hotel' | 'airline' = 'hotel',
  searchParams?: SearchParams
): Promise<string> {
  const params = new URLSearchParams();
  params.set('dateIndex', '7');

  if (storeType === 'hotel') {
    params.set('destination', searchParams?.destination || 'New York');
    params.set('adults', '2');
    params.set('rooms', '1');
  } else {
    params.set('origin', searchParams?.origin || 'JFK');
    params.set('destination', searchParams?.destination || 'LAX');
    params.set('adults', '1');
    params.set('children', '0');
    params.set('infants', '0');
  }

  await page.goto(`/products?${params.toString()}`);
  await page.waitForLoadState('networkidle');

  const productId = await openProductFromListing(page);
  await page.waitForTimeout(500);
  return productId;
}

export async function rapidViewProductViaFlow(
  page: Page,
  count: number,
  delayMs: number = 100,
  storeType: 'hotel' | 'airline' = 'hotel'
): Promise<string[]> {
  const productIds: string[] = [];

  for (let i = 0; i < count; i++) {
    const productId = await viewProductViaFlow(page, storeType);
    productIds.push(productId);

    await page.waitForTimeout(delayMs);
  }

  return productIds;
}

export async function humanLikeViewProduct(
  page: Page,
  storeType: 'hotel' | 'airline' = 'hotel'
): Promise<string> {
  const productId = await viewProductViaFlow(page, storeType);

  await page.hover('h1');
  await page.waitForTimeout(800 + Math.random() * 400);

  await page.mouse.wheel(0, 200);
  await page.waitForTimeout(500 + Math.random() * 300);

  const paragraphs = await page.locator('p').all();
  if (paragraphs.length > 0) {
    await paragraphs[0].hover();
    await page.waitForTimeout(600 + Math.random() * 400);
  }

  return productId;
}

export async function addToCart(page: Page): Promise<void> {
  const addBtn = page.locator('button:has-text("Add to Cart")');
  await addBtn.click();
  await page.waitForTimeout(500);
}
