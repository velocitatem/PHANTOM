import { Page } from '@playwright/test';

export async function navigateToProduct(
  page: Page,
  productId: string,
  storeType: 'hotel' | 'airline' = 'hotel'
): Promise<void> {
  await page.goto(`/${storeType}/products/${productId}`);
  await page.waitForLoadState('networkidle');
}

export async function rapidViewProduct(
  page: Page,
  productId: string,
  count: number,
  delayMs: number = 100,
  storeType: 'hotel' | 'airline' = 'hotel'
): Promise<void> {
  for (let i = 0; i < count; i++) {
    await navigateToProduct(page, productId, storeType);
    await page.waitForTimeout(delayMs);
  }
}

export async function humanLikeViewProduct(
  page: Page,
  productId: string,
  storeType: 'hotel' | 'airline' = 'hotel'
): Promise<void> {
  await navigateToProduct(page, productId, storeType);

  await page.hover('h1');
  await page.waitForTimeout(800 + Math.random() * 400);

  await page.mouse.wheel(0, 200);
  await page.waitForTimeout(500 + Math.random() * 300);

  const paragraphs = await page.locator('p').all();
  if (paragraphs.length > 0) {
    await paragraphs[0].hover();
    await page.waitForTimeout(600 + Math.random() * 400);
  }
}

export async function addToCart(page: Page): Promise<void> {
  const addBtn = page.locator('button:has-text("Add to Cart")');
  await addBtn.click();
  await page.waitForTimeout(500);
}

export async function getSessionId(page: Page): Promise<string | null> {
  const cookies = await page.context().cookies();
  const sessionCookie = cookies.find(c => c.name === 'phantom_session_id');
  return sessionCookie?.value || null;
}

export async function createFreshSession(page: Page): Promise<string> {
  await page.context().clearCookies();
  await page.goto('/');
  await page.waitForTimeout(500);

  const sid = await getSessionId(page);
  if (!sid) throw new Error('Session not created');
  return sid;
}
