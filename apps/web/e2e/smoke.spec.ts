import { expect, test } from '@playwright/test';

test('landing renders the hero and the learning path', async ({ page }) => {
  await page.goto('/');
  await expect(page.getByRole('heading', { level: 1 })).toContainText('See inside');
  await expect(page.getByRole('heading', { name: /One idea at a time/ })).toBeVisible();
  await expect(page.getByRole('heading', { name: 'Transformers & LLMs' })).toBeVisible();
});

test('theme selector switches and persists the palette', async ({ page }) => {
  await page.goto('/');
  await page.getByRole('radio', { name: 'Dark theme' }).click();
  await expect(page.locator('html')).toHaveAttribute('data-theme', 'dark');

  await page.reload();
  await expect(page.locator('html')).toHaveAttribute('data-theme', 'dark');

  await page.getByRole('radio', { name: 'Light theme' }).click();
  await expect(page.locator('html')).toHaveAttribute('data-theme', 'light');
});

test('design-system page is reachable via the router', async ({ page }) => {
  await page.goto('/');
  await page.getByRole('link', { name: 'Design system' }).click();
  await expect(page.getByRole('heading', { level: 1 })).toContainText('Parchment');
});

test('has no horizontal overflow on mobile', async ({ page }) => {
  await page.setViewportSize({ width: 375, height: 812 });
  await page.goto('/');
  const overflow = await page.evaluate(
    () => document.documentElement.scrollWidth - document.documentElement.clientWidth,
  );
  expect(overflow).toBeLessThanOrEqual(0);
});

test('footer shows credit text and LinkedIn / GitHub icon links (name is not a link)', async ({
  page,
}) => {
  await page.goto('/');
  const footer = page.getByRole('contentinfo');
  await expect(footer.getByText('Francisco Javier Gómez Pulido')).toBeVisible();
  await expect(footer.getByRole('link', { name: 'Francisco Javier Gómez Pulido' })).toHaveCount(0);
  await expect(footer.getByRole('link', { name: 'LinkedIn' })).toHaveAttribute(
    'href',
    'https://www.linkedin.com/in/frangomezpulido',
  );
  await expect(footer.getByRole('link', { name: 'GitHub profile' })).toHaveAttribute(
    'href',
    'https://github.com/fragompul',
  );
});
