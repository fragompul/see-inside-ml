import { expect, test } from '@playwright/test';

test.beforeEach(async ({ page }) => {
  page.on('pageerror', (error) => {
    throw error;
  });
  await page.goto('/#/classic/logistic-regression');
  await expect(page.getByRole('heading', { name: /Logistic regression/, level: 1 })).toBeVisible();
});

test('gradient descent moves the weights and Newton reports its iterations', async ({ page }) => {
  await expect(page.getByTestId('newton-steps')).toContainText(/^\d+$/);
  const start = await page.getByTestId('gd-state').textContent();
  await page.getByRole('button', { name: 'Play steps' }).click();
  await expect(page.getByTestId('gd-state')).not.toHaveText(start ?? '', { timeout: 15_000 });
});

test('the XOR experiments show a line failing and features fixing it', async ({ page }) => {
  await page.locator('button[data-experiment="xor-fails"]').click();
  await expect(page.getByTestId('train-accuracy')).toHaveText(/^[3-6]\d(\.\d)?%$/);
  await page.locator('button[data-experiment="xor-features"]').click();
  await expect(page.getByTestId('train-accuracy')).toHaveText(/^(9\d(\.\d)?|100)%$/);
});
