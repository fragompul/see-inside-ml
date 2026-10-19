import { expect, test } from '@playwright/test';

test.beforeEach(async ({ page }) => {
  page.on('pageerror', (error) => {
    throw error;
  });
  await page.goto('/#/gan');
  await expect(
    page.getByRole('heading', { name: /Generative adversarial networks/ }),
  ).toBeVisible();
});

test('shows the three sections, the duel and the statistics', async ({ page }) => {
  for (const title of ['Forger against detective', 'Is it working?', 'The rules of the game']) {
    await expect(page.getByRole('heading', { name: title })).toBeVisible();
  }
  await expect(page.getByRole('img', { name: /The plane where the generator/ })).toBeVisible();
  await expect(page.getByText('Clusters found')).toBeVisible();
  for (const chart of [
    'What the detective thinks',
    'Distance to the real data',
    'Losses',
    'Gradient to the generator',
  ]) {
    await expect(page.getByRole('img', { name: new RegExp(chart) })).toBeVisible();
  }
});

test('training advances the iteration count and can be paused', async ({ page }) => {
  await expect(page.getByText('0 iterations')).toBeVisible();
  await page.getByRole('button', { name: 'Train the network' }).click();
  await expect(page.getByRole('button', { name: 'Pause training' })).toBeVisible();
  await expect(page.getByText(/^[1-9]\d* iterations$/)).toBeVisible({ timeout: 15_000 });
  await page.getByRole('button', { name: 'Pause training' }).click();
  await page.getByRole('button', { name: 'Reset the networks' }).click();
  await expect(page.getByText('0 iterations')).toBeVisible();
});

test('ten steps run exactly ten iterations', async ({ page }) => {
  await page.getByRole('button', { name: 'Run 10 steps' }).click();
  await expect(page.getByText('10 iterations')).toBeVisible();
  await expect(page.getByText('D says real (on real)')).toBeVisible();
});

test('changing the dataset restarts the duel', async ({ page }) => {
  await page.getByRole('button', { name: 'Run 10 steps' }).click();
  await expect(page.getByText('10 iterations')).toBeVisible();
  await page.getByRole('radio', { name: 'Ring' }).click();
  await expect(page.getByText('0 iterations')).toBeVisible();
  await expect(page.getByText('Clusters found')).toBeHidden(); // only meaningful for eight blobs
});

test('switching the generator loss changes the formula', async ({ page }) => {
  await expect(page.getByText(/Maximise log D\(G\(z\)\)/)).toBeVisible();
  await page.getByRole('radio', { name: 'Minimax' }).click();
  await expect(page.getByText(/the original formulation/i)).toBeVisible();
});

test('an experiment sets the lab up and trains it', async ({ page }) => {
  await page.getByRole('button', { name: 'Mode collapse' }).click();
  await expect(page.getByText(/What to look for/)).toBeVisible();
  await expect(page.getByRole('button', { name: 'Pause training' })).toBeVisible();
  await expect(page.getByText(/^[1-9]\d* iterations$/)).toBeVisible({ timeout: 15_000 });
  await page.getByRole('button', { name: 'Pause training' }).click();
  await page.getByRole('radio', { name: 'Ring' }).click();
  await expect(page.getByText(/What to look for/)).toBeHidden();
});

test('fits a phone screen without horizontal page scroll', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 800 });
  await page.reload();
  await expect(
    page.getByRole('heading', { name: /Generative adversarial networks/ }),
  ).toBeVisible();
  const overflow = await page.evaluate(
    () => document.documentElement.scrollWidth - document.documentElement.clientWidth,
  );
  expect(overflow).toBeLessThanOrEqual(0);
});
