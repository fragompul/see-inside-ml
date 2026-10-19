import { expect, test } from '@playwright/test';

test.beforeEach(async ({ page }) => {
  page.on('pageerror', (error) => {
    throw error;
  });
  await page.goto('/#/rnn');
  await expect(page.getByRole('heading', { name: /Recurrent networks/ })).toBeVisible();
});

test('shows the four sections and the default network', async ({ page }) => {
  for (const title of [
    'A task and a sequence',
    'The network unrolled in time',
    'Why long memory is hard',
    'Train it',
  ]) {
    await expect(page.getByRole('heading', { name: title })).toBeVisible();
  }
  await expect(page.getByText('LSTM', { exact: true }).first()).toBeVisible();
  await expect(page.getByRole('list', { name: 'Input sequence' })).toBeVisible();
  await expect(page.getByRole('img', { name: /Forget gate/ })).toBeVisible();
});

test('editing a token changes the rule answer', async ({ page }) => {
  await page.getByRole('radio', { name: 'Remember the first' }).click();
  const answer = page.getByText(/The rule says/);
  const before = await answer.textContent();
  await page.getByRole('button', { name: /^Token 1:/ }).click();
  await expect(answer).not.toHaveText(before ?? '');
  await expect(page.getByText(/· edited/)).toBeVisible();
});

test('switching the cell shows its own gates', async ({ page }) => {
  await page.getByRole('radio', { name: 'RNN', exact: true }).click();
  await expect(page.getByRole('img', { name: /Forget gate/ })).toBeHidden();
  await expect(page.getByText(/vanilla RNN has no gates/i)).toBeVisible();
  await page.getByRole('radio', { name: 'GRU', exact: true }).click();
  await expect(page.getByRole('img', { name: /Update gate/ })).toBeVisible();
});

test('the unrolled view reacts to the pointer', async ({ page }) => {
  await page.getByRole('button', { name: /^Step 1, token/ }).hover();
  await expect(page.getByText('Step 1', { exact: true })).toBeVisible();
  await page.getByRole('button', { name: /^Step 3, token/ }).hover();
  await expect(page.getByText('Step 3', { exact: true })).toBeVisible();
});

test('an experiment configures the lab, trains it and can be left', async ({ page }) => {
  await page.getByRole('button', { name: 'Gates fix it: LSTM' }).click();
  await expect(page.getByText(/What to look for/)).toBeVisible();
  await expect(page.getByRole('button', { name: 'Pause training' })).toBeVisible();
  await expect(page.getByText(/test accuracy/)).toBeVisible({ timeout: 20_000 });
  await page.getByRole('button', { name: 'Pause training' }).click();
  await page.getByRole('radio', { name: 'GRU', exact: true }).click();
  await expect(page.getByText(/What to look for/)).toBeHidden(); // no longer that experiment
});

test('measures accuracy at other lengths after training', async ({ page }) => {
  await expect(page.getByRole('button', { name: 'Measure other lengths' })).toBeDisabled();
  await page.getByRole('button', { name: 'Train one epoch' }).click();
  await page.getByRole('button', { name: 'Measure other lengths' }).click();
  await expect(page.getByRole('img', { name: 'Accuracy for each sequence length' })).toBeVisible();
});

test('fits a phone screen without horizontal page scroll', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 800 });
  await page.reload();
  await expect(page.getByRole('heading', { name: /Recurrent networks/ })).toBeVisible();
  const overflow = await page.evaluate(
    () => document.documentElement.scrollWidth - document.documentElement.clientWidth,
  );
  expect(overflow).toBeLessThanOrEqual(0);
});
