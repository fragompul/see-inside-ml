import { expect, test, type Page } from '@playwright/test';

test.beforeEach(async ({ page }) => {
  page.on('pageerror', (error) => {
    throw error;
  });
  await page.goto('/#/rnn/writer');
  await expect(
    page.getByRole('heading', { name: /A network that writes/, level: 1 }),
  ).toBeVisible();
});

const steps = async (page: Page) => Number(await page.getByTestId('steps').innerText());

test('training lowers the loss and the network starts to write', async ({ page }) => {
  const before = Number(await page.getByTestId('loss').innerText());
  await page.getByRole('button', { name: 'Train' }).click();
  await expect.poll(() => steps(page), { timeout: 30_000 }).toBeGreaterThan(60);
  await page.getByRole('button', { name: 'Pause' }).click();
  const after = Number(await page.getByTestId('loss').innerText());
  expect(after).toBeLessThan(before);
  await expect(page.getByTestId('sample')).not.toBeEmpty();
});

test('start again resets the network', async ({ page }) => {
  await page.getByRole('button', { name: 'Train' }).click();
  await expect.poll(() => steps(page), { timeout: 30_000 }).toBeGreaterThan(10);
  await page.getByRole('button', { name: 'Start again' }).click();
  await expect.poll(() => steps(page)).toBe(0);
});

test('an experiment trains the network to its target and stops', async ({ page }) => {
  await page.locator('button[data-experiment="memorise-rhyme"]').click();
  await expect(page.getByText('What to look for:')).toBeVisible();
  await expect.poll(() => steps(page), { timeout: 60_000 }).toBe(150);
  await expect(page.getByTestId('sample')).toContainText(/wonder/);
  await expect(page.getByRole('button', { name: 'Train' })).toBeVisible();
});
