import { expect, test, type Page } from '@playwright/test';

const epoch = async (page: Page) => Number(await page.getByTestId('epoch').textContent());

test.beforeEach(async ({ page }) => {
  page.on('pageerror', (error) => {
    throw error;
  });
  await page.goto('/#/mlp');
  await expect(page.getByRole('button', { name: 'Train the network' })).toBeVisible();
});

test('trains in a worker, advances epochs and pauses', async ({ page }) => {
  expect(await epoch(page)).toBe(0);
  await page.getByRole('button', { name: 'Train the network' }).click();
  await expect.poll(() => epoch(page)).toBeGreaterThan(20);

  await page.getByRole('button', { name: 'Pause training' }).click();
  await page.waitForTimeout(250); // let the in-flight batch land
  const frozen = await epoch(page);
  await page.waitForTimeout(400);
  expect(await epoch(page)).toBe(frozen);
});

test('step advances one epoch; reset returns to zero', async ({ page }) => {
  await page.getByRole('button', { name: 'Step one epoch' }).click();
  await expect.poll(() => epoch(page)).toBe(1);
  await page.getByRole('button', { name: 'Step one epoch' }).click();
  await expect.poll(() => epoch(page)).toBe(2);
  await page.getByRole('button', { name: 'Reset weights' }).click();
  await expect.poll(() => epoch(page)).toBe(0);
});

test('keyboard shortcuts drive training', async ({ page }) => {
  await page.locator('h1').click(); // focus the page body, away from any control
  await page.keyboard.press('Space');
  // generous timeout: the worker slows down when the whole suite runs on a busy machine
  await expect.poll(() => epoch(page), { timeout: 30_000 }).toBeGreaterThan(5);
  await page.keyboard.press('Space');
  await expect(page.getByRole('button', { name: 'Train the network' })).toBeVisible();
});

test('hovering a neuron explains it; hovering the plane probes the model', async ({ page }) => {
  await page.getByRole('button', { name: 'L1 · n1' }).hover();
  await expect(page.getByText('Hidden 1 · neuron 1')).toBeVisible();
  await expect(page.getByText('incoming weights').or(page.getByText('bias b'))).toBeVisible();

  const plane = page.getByTestId('decision-plane');
  const box = await plane.boundingBox();
  if (!box) throw new Error('no canvas');
  await page.mouse.move(box.x + 8, box.y + 8);
  await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2);
  await expect(page.getByText(/class 0/)).toBeVisible();
});

test('experiments load a lesson and start training', async ({ page }) => {
  await page.getByRole('button', { name: 'Feature engineering' }).click();
  await expect(page.getByText(/What to look for/)).toBeVisible();
  await expect(page.getByText(/x₁·x₂/).first()).toBeVisible();
  await expect.poll(() => epoch(page)).toBeGreaterThan(10);
  await expect(page.getByRole('button', { name: 'Pause training' })).toBeVisible();
});

test('editing the architecture rebuilds the network', async ({ page }) => {
  await expect(page.getByRole('button', { name: 'L2 · n3' })).toBeVisible();
  await page.getByRole('button', { name: 'Add a hidden layer' }).click();
  await expect(page.getByRole('button', { name: 'L3 · n3' })).toBeVisible();
  await page.getByRole('button', { name: 'Add a neuron to hidden layer 1' }).click();
  await expect(page.getByRole('button', { name: 'L1 · n5' })).toBeVisible();
  await page.getByRole('button', { name: 'Remove hidden layer 3' }).click();
  await expect(page.getByRole('button', { name: 'L3 · n3' })).toHaveCount(0);
});

test('the setup lives in the URL and survives a reload', async ({ page }) => {
  await page.getByRole('button', { name: 'Spiral', exact: true }).click();
  await expect(page).toHaveURL(/[?&]c=/);
  await page.reload();
  await expect(page.getByRole('button', { name: 'Spiral', exact: true })).toHaveAttribute(
    'aria-pressed',
    'true',
  );
});

test('a corrupted share link falls back to the defaults', async ({ page }) => {
  await page.goto('/#/mlp?c=@@not-valid@@');
  await page.reload();
  await expect(page.getByRole('button', { name: 'Train the network' })).toBeVisible();
  await expect(page.getByRole('button', { name: 'Circles' })).toHaveAttribute(
    'aria-pressed',
    'true',
  );
});

test('switching theme keeps the plane and network rendered', async ({ page }) => {
  await page.getByRole('radio', { name: 'Light theme' }).click();
  await expect(page.locator('html')).toHaveAttribute('data-theme', 'light');
  await page.getByRole('radio', { name: 'Dark theme' }).click();
  await expect(page.locator('html')).toHaveAttribute('data-theme', 'dark');
  await expect(page.getByRole('button', { name: 'L1 · n1' })).toBeVisible();
});

test('fits a phone screen without horizontal page scroll', async ({ page }) => {
  await page.setViewportSize({ width: 375, height: 812 });
  await page.goto('/#/mlp');
  await expect(page.getByRole('button', { name: 'Train the network' })).toBeVisible();
  const overflow = await page.evaluate(
    () => document.documentElement.scrollWidth - document.documentElement.clientWidth,
  );
  expect(overflow).toBeLessThanOrEqual(0);
});

test('the step inspector walks through forward, loss, backward and update', async ({ page }) => {
  await page.getByRole('button', { name: 'Inspect a step' }).click();
  await expect(page.getByRole('heading', { name: 'Anatomy of one step' })).toBeVisible();
  await expect(page.getByRole('heading', { name: '1. Forward pass' })).toBeVisible();

  await page.getByRole('button', { name: 'Next stage' }).click();
  await expect(page.getByRole('heading', { name: '2. Loss' })).toBeVisible();
  await page.getByRole('button', { name: /Backward/ }).click();
  await expect(page.getByRole('heading', { name: '3. Backward pass' })).toBeVisible();
  await page.getByRole('button', { name: /Update/ }).click();
  await expect(page.getByRole('heading', { name: '4. Weight update' })).toBeVisible();
  await expect(page.getByText(/Biggest move/)).toBeVisible();

  // Hovering an edge in the inspector shows its gradient maths.
  await page.getByRole('button', { name: 'L1 · n1' }).hover();
  await expect(page.getByText('In this step')).toBeVisible();

  // Starting to train closes the inspector.
  await page.getByRole('button', { name: 'Train the network' }).click();
  await expect(page.getByRole('heading', { name: 'Anatomy of one step' })).toHaveCount(0);
});

test('the loss landscape and the weight histograms follow training', async ({ page }) => {
  await expect(page.getByRole('heading', { name: 'Loss landscape' })).toBeVisible();
  await expect(page.getByRole('heading', { name: 'Weights and gradients' })).toBeVisible();
  await expect(page.getByTestId('landscape-readout')).toContainText(/loss now|computing/);
  await expect(page.getByTestId('hist-layer-0')).toBeVisible();

  await page.getByRole('button', { name: 'Train the network' }).click();
  await expect.poll(() => epoch(page)).toBeGreaterThan(30);
  await page.getByRole('button', { name: 'Pause training' }).click();
  // after a pause the slice is refreshed for the trained weights and shows a real loss value
  const map = page.getByRole('img', { name: /Map of the training loss/ });
  await expect(map).toBeVisible();
  await expect(page.getByTestId('landscape-readout')).toContainText(/loss now \d/);
  await map.hover({ position: { x: 100, y: 100 } });
  await expect(page.getByTestId('landscape-readout')).toContainText(/loss \d.*at \(/);
  await page.getByText('Random directions', { exact: true }).click();
  await expect(page.getByTestId('landscape-readout')).toContainText(/loss now \d/);
});

test('points can be added by hand, training carries on, and the data can be restored', async ({
  page,
}) => {
  await page.getByRole('button', { name: 'Step one epoch' }).click();
  await expect.poll(() => epoch(page)).toBe(1);
  await expect(page.getByRole('button', { name: 'Restore the data' })).toHaveCount(0);

  await page.getByRole('radio', { name: 'Add', exact: true }).click();
  await page.getByRole('button', { name: 'Add class 1' }).click();
  const plane = page.getByTestId('decision-plane');
  const box = (await plane.boundingBox())!;
  await page.mouse.click(box.x + box.width * 0.5, box.y + box.height * 0.5);
  await expect(page.getByRole('button', { name: 'Restore the data' })).toBeVisible();
  // the epoch counter was not reset by the edit
  expect(await epoch(page)).toBe(1);

  await page.getByRole('radio', { name: 'Erase', exact: true }).click();
  await page.mouse.click(box.x + box.width * 0.5, box.y + box.height * 0.5);
  await page.getByRole('radio', { name: 'Look', exact: true }).click();
  await page.getByRole('button', { name: 'Restore the data' }).click();
  await expect(page.getByRole('button', { name: 'Restore the data' })).toHaveCount(0);
  await page.getByRole('button', { name: 'Step one epoch' }).click();
  await expect.poll(() => epoch(page)).toBe(2);
});
