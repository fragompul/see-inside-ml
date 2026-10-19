import { expect, test, type Page } from '@playwright/test';

test.beforeEach(async ({ page }) => {
  page.on('pageerror', (error) => {
    throw error;
  });
  await page.goto('/#/rl');
  await expect(page.getByRole('heading', { name: /Reinforcement learning/ })).toBeVisible();
});

const episodes = (page: Page) => page.getByTestId('episodes');

test('shows the three sections, the gridworld and the cliff by default', async ({ page }) => {
  for (const title of [
    'A world to explore',
    'Learning by trial and error',
    'The rules of learning',
  ]) {
    await expect(page.getByRole('heading', { name: title })).toBeVisible();
  }
  await expect(page.getByRole('img', { name: /The gridworld/ })).toBeVisible();
  await expect(page.getByText('12×4 grid')).toBeVisible();
  await expect(page.getByText('Q-learning', { exact: true }).first()).toBeVisible();
});

test('fast-forwarding episodes fills the statistics and the optimal-move share', async ({
  page,
}) => {
  await expect(episodes(page)).toHaveText('0');
  await page.getByRole('button', { name: 'Run 100 episodes' }).click();
  await expect(episodes(page)).toHaveText('100');
  await expect(page.getByTestId('agreement')).toContainText('%');
  await page.getByRole('button', { name: 'Run 100 episodes' }).click();
  await expect(episodes(page)).toHaveText('200');
});

test('a single step moves the step counter, and reset forgets everything', async ({ page }) => {
  await page.getByRole('button', { name: 'Take one step' }).click();
  await expect(page.getByTestId('steps')).toHaveText('1');
  await page.getByRole('button', { name: 'Run 100 episodes' }).click();
  await page.getByRole('button', { name: 'Reset the agent' }).click();
  await expect(episodes(page)).toHaveText('0');
  await expect(page.getByTestId('steps')).toHaveText('0');
});

test('training runs continuously and can be paused', async ({ page }) => {
  await page.getByRole('button', { name: 'Train the network' }).click();
  await expect(page.getByRole('button', { name: 'Pause training' })).toBeVisible();
  await expect.poll(async () => Number(await episodes(page).textContent())).toBeGreaterThan(5);
  await page.getByRole('button', { name: 'Pause training' }).click();
  await expect(page.getByRole('button', { name: 'Train the network' })).toBeVisible();
});

test('presets change the layout and restart learning', async ({ page }) => {
  await page.getByRole('button', { name: 'Run 100 episodes' }).click();
  await expect(episodes(page)).toHaveText('100');
  await page.getByRole('radio', { name: 'Maze' }).click();
  await expect(page.getByText('8×6 grid')).toBeVisible();
  await expect(episodes(page)).toHaveText('0');
});

test('painting a wall restarts learning; the inspector shows a cell', async ({ page }) => {
  await page.getByRole('button', { name: 'Run 100 episodes' }).click();
  await expect(episodes(page)).toHaveText('100');
  await page.getByRole('radio', { name: 'Wall' }).click();
  const grid = page.getByRole('img', { name: /The gridworld/ });
  await grid.scrollIntoViewIfNeeded();
  const box = (await grid.boundingBox())!;
  // the cliff is 12 × 4: cell (1, 1) sits at 1.5/12 of the width and 1.5/4 of the height
  await page.mouse.click(box.x + box.width * (1.5 / 12), box.y + box.height * (1.5 / 4));
  await expect(episodes(page)).toHaveText('0');
  await page.getByRole('radio', { name: 'Inspect' }).click();
  await grid.scrollIntoViewIfNeeded();
  const again = (await grid.boundingBox())!;
  await page.mouse.click(again.x + again.width * (2.5 / 12), again.y + again.height * (1.5 / 4));
  await expect(page.getByText(/Cell \(2, 1\)/)).toBeVisible();
});

test('the layout cannot be left without a goal', async ({ page }) => {
  await page.getByRole('radio', { name: 'Erase' }).click();
  const grid = page.getByRole('img', { name: /The gridworld/ });
  await grid.scrollIntoViewIfNeeded();
  const box = (await grid.boundingBox())!;
  await page.mouse.click(box.x + box.width * (11.5 / 12), box.y + box.height * (3.5 / 4)); // the only goal
  await page.getByRole('button', { name: 'Run 100 episodes' }).scrollIntoViewIfNeeded();
  await expect(page.getByRole('alert')).toContainText(/at least one goal/);
});

test('switching the algorithm changes the update rule', async ({ page }) => {
  await page.getByRole('radio', { name: 'SARSA', exact: true }).click();
  await expect(page.getByText(/learns the value of what it really does/i)).toBeVisible();
  await page.getByRole('radio', { name: 'Expected SARSA' }).click();
  await expect(page.getByText(/Averages over what it might do next/)).toBeVisible();
});

test('an experiment sets the lab up and trains it', async ({ page }) => {
  await page.getByRole('button', { name: 'SARSA takes the safe road' }).click();
  await expect(page.getByText(/What to look for/)).toBeVisible();
  await expect(page.getByRole('button', { name: 'Pause training' })).toBeVisible();
  await expect.poll(async () => Number(await episodes(page).textContent())).toBeGreaterThan(20);
  await page.getByRole('button', { name: 'Pause training' }).click();
  await page.getByRole('radio', { name: 'Q-learning', exact: true }).click();
  await expect(page.getByText(/What to look for/)).toBeHidden();
});

test('fits a phone screen without horizontal page scroll', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 800 });
  await page.reload();
  await expect(page.getByRole('heading', { name: /Reinforcement learning/ })).toBeVisible();
  const overflow = await page.evaluate(
    () => document.documentElement.scrollWidth - document.documentElement.clientWidth,
  );
  expect(overflow).toBeLessThanOrEqual(0);
});
