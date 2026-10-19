import { expect, test } from '@playwright/test';

test.beforeEach(async ({ page }) => {
  page.on('pageerror', (error) => {
    throw error;
  });
  await page.goto('/#/classic/linear-regression');
  await expect(page.getByRole('heading', { name: /Linear regression/ })).toBeVisible();
});

test('shows the fit, its metrics and the extra panels', async ({ page }) => {
  for (const title of [
    'The data and the fitted curve',
    'Descending to the line',
    'Diagnostics: is the fit trustworthy?',
    'Anscombe’s quartet',
  ]) {
    await expect(page.getByRole('heading', { name: title })).toBeVisible();
  }
  await expect(page.getByTestId('train-r2')).toContainText(/\d\.\d{3}/);
  await expect(page.getByTestId('lr-summary')).toContainText(/R² = /);
});

test('an experiment changes the data and the fit', async ({ page }) => {
  const before = await page.getByTestId('train-r2').textContent();
  await page.getByRole('button', { name: /A line on a parabola/ }).click();
  await expect(page.getByTestId('train-r2')).not.toHaveText(before ?? '');
});

test('gradient descent steps towards the closed-form line', async ({ page }) => {
  const start = await page.getByTestId('gd-line').textContent();
  await page.getByRole('button', { name: 'Play steps' }).click();
  await expect(page.getByTestId('gd-line')).not.toHaveText(start ?? '', { timeout: 15_000 });
});

test('loading an Anscombe dataset replaces the points', async ({ page }) => {
  await page.getByRole('button', { name: 'Anscombe II', exact: true }).click();
  await expect(page.getByText('11 points')).toBeVisible();
  await expect(page.getByText('edited by hand')).toBeVisible();
});
