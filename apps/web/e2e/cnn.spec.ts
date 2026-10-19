import { expect, test } from '@playwright/test';

test.beforeEach(async ({ page }) => {
  page.on('pageerror', (error) => {
    throw error;
  });
  await page.goto('/#/cnn');
  await expect(page.getByRole('heading', { name: /Convolutional networks/ })).toBeVisible();
});

test('shows all four sections and the network anatomy', async ({ page }) => {
  for (const title of [
    'Draw a digit, read its answer',
    'Convolution by hand',
    'What each layer sees',
    'Train it',
  ]) {
    await expect(page.getByRole('heading', { name: title })).toBeVisible();
  }
  await expect(page.getByText('6 + 12 filters')).toBeVisible();
  await expect(page.getByRole('list', { name: 'Probability of each digit' })).toBeVisible();
});

test('drawing a stroke replaces the sample, clearing empties the pad', async ({ page }) => {
  await expect(page.getByText(/This test image is a/)).toBeVisible();
  const pad = page.getByRole('img', { name: /Drawing pad/ });
  const box = (await pad.boundingBox())!;
  await page.mouse.move(box.x + box.width * 0.3, box.y + box.height * 0.2);
  await page.mouse.down();
  await page.mouse.move(box.x + box.width * 0.5, box.y + box.height * 0.8, { steps: 8 });
  await page.mouse.up();
  await expect(page.getByText(/This test image is a/)).toBeHidden();
  await expect(page.getByText(/sure/)).toBeVisible();

  await page.getByRole('button', { name: 'Clear' }).click();
  await expect(page.getByText('Nothing to read yet')).toBeVisible();
});

test('the convolution window computes the sum of products', async ({ page }) => {
  const sum = page.getByTestId('conv-sum');
  await page.getByLabel('Kernel', { exact: true }).selectOption('identity');
  const before = await sum.textContent();
  await page.getByRole('slider', { name: 'Window column' }).focus();
  await page.keyboard.press('ArrowRight');
  await page.keyboard.press('ArrowRight');
  await page.keyboard.press('ArrowRight');
  // the identity kernel returns the centre pixel, which differs from one column to the next
  await expect(sum).not.toHaveText(before ?? '');

  await page.getByLabel('Kernel weight row 2, column 2').fill('0');
  await expect(sum).toHaveText('0.00');
});

test('training improves accuracy and can be paused', async ({ page }) => {
  await page.getByRole('button', { name: 'Train the network' }).click();
  await expect(page.getByRole('button', { name: 'Pause training' })).toBeVisible();
  await expect(page.getByText(/test accuracy/)).toBeVisible({ timeout: 20_000 });
  await page.getByRole('button', { name: 'Pause training' }).click();
  await expect(page.getByRole('button', { name: 'Train the network' })).toBeVisible();
  const epochs = page.getByText(/\d+ epochs trained/);
  await expect(epochs).not.toHaveText('0 epochs trained');

  await page.getByRole('button', { name: 'Reset the network' }).click();
  await expect(page.getByText('0 epochs trained')).toBeVisible();
});

test('changing the architecture rebuilds the network', async ({ page }) => {
  await page.getByRole('radio', { name: '5 × 5' }).click();
  await expect(page.getByText('5×5 kernels')).toBeVisible();
  await expect(page.getByText('6 filters of 5×5')).toBeVisible();
});

test('shows where the network looks with two maps of the evidence', async ({ page }) => {
  await expect(page.getByRole('heading', { name: 'Where does it look?' })).toBeVisible();
  await expect(page.getByRole('img', { name: /evidence map/i })).toBeVisible();
  await expect(page.getByTestId('attention-note')).toContainText('adds up to the score exactly');
  await page.getByLabel('Evidence for').selectOption('3');
  await expect(page.getByText(/where the score for 3 comes from/)).toBeVisible();
});

test('attacking a digit reports whether it was fooled, and robustness can be measured', async ({
  page,
}) => {
  await expect(page.getByRole('heading', { name: 'Fool the network' })).toBeVisible();
  await expect(page.getByTestId('adv-verdict')).toHaveText(/Fooled!|Still right/);
  await page.getByText('PGD (12 steps)').click();
  const slider = page.getByRole('slider', { name: 'Budget ε per pixel' });
  await slider.focus();
  await page.keyboard.press('End');
  await expect(page.getByTestId('adv-answer')).toContainText('attacked →');
  await page.getByRole('button', { name: 'Test robustness' }).click();
  await expect(page.getByText('Accuracy under attack')).toBeVisible({ timeout: 30_000 });
});
