import { expect, test } from '@playwright/test';

test.beforeEach(async ({ page }) => {
  page.on('pageerror', (error) => {
    throw error;
  });
  await page.goto('/#/foundations');
  await expect(page.getByRole('heading', { name: /neurons and learning/ })).toBeVisible();
});

test('shows the four sections', async ({ page }) => {
  for (const title of [
    'One neuron, one line',
    'Activation functions',
    'Learning is rolling downhill',
    'What one neuron cannot do',
  ]) {
    await expect(page.getByRole('heading', { name: title })).toBeVisible();
  }
  await expect(page.getByRole('img', { name: /The plane with the data points/ })).toBeVisible();
  await expect(page.getByRole('img', { name: /The loss landscape/ })).toBeVisible();
});

test('the weights change how many points the line gets right', async ({ page }) => {
  const accuracy = page.getByTestId('accuracy');
  await expect(accuracy).toContainText('100%');
  await page.getByRole('slider', { name: 'Bias b', exact: true }).focus();
  await page.keyboard.press('End');
  await expect(accuracy).not.toContainText('100%');
});

test('the best line solves AND but cannot solve XOR', async ({ page }) => {
  await page.getByRole('radio', { name: 'AND', exact: true }).first().click();
  await page.getByRole('button', { name: 'Find the best line' }).click();
  await expect(page.getByTestId('best-result')).toContainText('separates every point');
  await expect(page.getByTestId('accuracy')).toContainText('100%');
  await page.getByRole('radio', { name: 'XOR', exact: true }).first().click();
  await page.getByRole('button', { name: 'Find the best line' }).click();
  await expect(page.getByTestId('best-result')).toContainText('75%');
});

test('clicking a point shows its arithmetic', async ({ page }) => {
  await page.getByRole('radio', { name: 'XOR', exact: true }).first().click();
  const plane = page.getByRole('img', { name: /The plane with the data points/ });
  await plane.scrollIntoViewIfNeeded();
  const box = (await plane.boundingBox())!;
  // x from -0.6 to 1.6 across the canvas: the point (1, 1) is at 72.7% of the width and 27.3% of the height
  await page.mouse.click(box.x + box.width * 0.727, box.y + box.height * 0.273);
  await expect(page.getByText(/point \(1\.00, 1\.00\)/)).toBeVisible();
});

test('the activation gallery reports the steepest slope', async ({ page }) => {
  await expect(page.getByTestId('max-slope')).toHaveText('0.25'); // sigmoid is the default
  await page.getByRole('radio', { name: 'ReLU' }).last().click();
  await expect(page.getByTestId('max-slope')).toHaveText('1.00');
  await expect(page.getByTestId('ten-layers')).toHaveText('1.00');
});

test('the ball rolls downhill, pauses and goes back to the start', async ({ page }) => {
  const loss = page.getByTestId('loss');
  const before = Number(await loss.textContent());
  await page.getByRole('button', { name: 'Roll the ball' }).click();
  await expect(page.getByRole('button', { name: 'Pause the ball' })).toBeVisible();
  await expect
    .poll(async () => Number(await loss.textContent()), { timeout: 15_000 })
    .toBeLessThan(before);
  await page.getByRole('button', { name: 'Pause the ball' }).click();
  await page.getByRole('button', { name: 'Back to start' }).click();
  await expect(loss).toHaveText(String(before.toFixed(3)));
});

test('dropping the ball elsewhere moves its start', async ({ page }) => {
  const land = page.getByRole('img', { name: /The loss landscape/ });
  await land.scrollIntoViewIfNeeded();
  const box = (await land.boundingBox())!;
  await page.mouse.click(box.x + box.width * 0.75, box.y + box.height * 0.5);
  await expect(page.getByTestId('step')).toHaveText('0');
  await expect(page.getByTestId('weights')).toHaveText(/^[34]\.\d, [-0]\.\d$|^[34]\.\d, -?0\.\d$/);
});

test('experiments set up the descent', async ({ page }) => {
  await page.getByRole('button', { name: 'No line solves XOR' }).click();
  await expect(page.getByText(/What to look for/)).toBeVisible();
  await expect(page.getByRole('radio', { name: 'XOR', exact: true }).last()).toHaveAttribute(
    'aria-checked',
    'true',
  );
  await page.getByRole('button', { name: 'Pause the ball' }).click();
  await page.getByRole('radio', { name: 'Momentum' }).click();
  await expect(page.getByText(/What to look for/)).toBeHidden();
});

test('links to the MLP playground', async ({ page }) => {
  await page.getByRole('link', { name: /Open the MLP playground/ }).click();
  await expect(page).toHaveURL(/#\/mlp$/);
});

test('fits a phone screen without horizontal page scroll', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 800 });
  await page.reload();
  await expect(page.getByRole('heading', { name: /neurons and learning/ })).toBeVisible();
  const overflow = await page.evaluate(
    () => document.documentElement.scrollWidth - document.documentElement.clientWidth,
  );
  expect(overflow).toBeLessThanOrEqual(0);
});
