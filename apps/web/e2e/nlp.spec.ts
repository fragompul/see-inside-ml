import { expect, test } from '@playwright/test';

test.beforeEach(async ({ page }) => {
  page.on('pageerror', (error) => {
    throw error;
  });
  await page.goto('/#/nlp');
  await expect(page.getByRole('heading', { name: /Embeddings & language/ })).toBeVisible();
});

test('shows the five sections, the map and the default numbers', async ({ page }) => {
  for (const title of [
    'From text to training pairs',
    'The map of meaning',
    'Nearest neighbours',
    'Arithmetic with meaning',
    'Train it',
  ]) {
    await expect(page.getByRole('heading', { name: title })).toBeVisible();
  }
  await expect(page.getByText('16-dimensional vectors')).toBeVisible();
  await expect(page.getByRole('group', { name: /Map of the word vectors/ }).first()).toBeVisible();
  await expect(page.getByRole('list', { name: 'Word families' }).first()).toBeVisible();
});

test('a sentence becomes pairs around the chosen centre word', async ({ page }) => {
  const pairs = page.getByRole('list', { name: 'Positive training pairs' });
  await expect(pairs.getByRole('listitem').first()).toBeVisible();
  await page.getByRole('button', { name: /^Word 1:/ }).click();
  await expect(page.getByRole('button', { name: /^Word 1:/ })).toHaveAttribute(
    'aria-pressed',
    'true',
  );
  const before = await page.getByText(/^sentence \d+ \/ \d+$/).textContent();
  await page.getByRole('button', { name: 'Next sentence' }).click();
  await expect(page.getByText(/^sentence \d+ \/ \d+$/)).not.toHaveText(before ?? '');
  await expect(
    page.getByRole('list', { name: 'Random negative words' }).getByRole('listitem'),
  ).toHaveCount(5);
});

test('clicking a word on the map selects it and lists its neighbours', async ({ page }) => {
  await page.getByRole('button', { name: 'Select cat' }).first().click();
  await expect(page.getByRole('list', { name: 'Words closest to cat' })).toBeVisible();
  await expect(page.getByRole('list', { name: 'Nearest words to cat' })).toBeVisible();
});

test('the similarity of a word with itself is one', async ({ page }) => {
  await page.getByRole('button', { name: 'Select paris' }).first().click();
  await page.getByLabel('Second').selectOption('paris');
  await expect(page.getByTestId('pair-similarity')).toHaveText('1.00');
  await page.getByLabel('Second').selectOption('cat');
  await expect(page.getByTestId('pair-similarity')).not.toHaveText('1.00');
});

test('the analogy calculator answers and never repeats its inputs', async ({ page }) => {
  await page.getByRole('button', { name: 'france : paris :: italy : ?' }).click();
  const answer = page.getByTestId('analogy-answer');
  await expect(answer).not.toHaveText('–');
  const text = (await answer.textContent()) ?? '';
  expect(['france', 'paris', 'italy']).not.toContain(text);
  await expect(page.getByRole('list', { name: 'Analogy accuracy by kind' })).toBeVisible();
});

test('training advances the steps and can be paused or reset', async ({ page }) => {
  await page.getByRole('button', { name: 'Train 25 steps' }).click();
  await expect(page.getByText('25 steps', { exact: true }).first()).toBeVisible();
  await page.getByRole('button', { name: 'Train the network' }).click();
  await expect(page.getByRole('button', { name: 'Pause training' })).toBeVisible();
  await page.getByRole('button', { name: 'Pause training' }).click();
  await page.getByRole('button', { name: 'Reset the vectors' }).click();
  await expect(page.getByText('0 steps', { exact: true }).first()).toBeVisible();
});

test('a far too high learning rate stops with an explanation', async ({ page }) => {
  await page.getByRole('button', { name: 'Learning rate too high' }).click();
  await expect(page.getByRole('alert')).toContainText(/learning rate is too high/, {
    timeout: 20_000,
  });
  await expect(page.getByRole('button', { name: 'Train the network' })).toBeVisible();
});

test('the experiments set the lab up', async ({ page }) => {
  await page.getByRole('button', { name: 'A narrow window' }).click();
  await expect(page.getByText(/What to look for/)).toBeVisible();
  await expect(page.getByText('window 1')).toBeVisible();
  await page.getByRole('button', { name: 'Pause training' }).click();
});

test('fits a phone screen without horizontal page scroll', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 800 });
  await page.reload();
  await expect(page.getByRole('heading', { name: /Embeddings & language/ })).toBeVisible();
  const overflow = await page.evaluate(
    () => document.documentElement.scrollWidth - document.documentElement.clientWidth,
  );
  expect(overflow).toBeLessThanOrEqual(0);
});
