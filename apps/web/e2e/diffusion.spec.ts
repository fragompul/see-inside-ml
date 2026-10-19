import { expect, test } from '@playwright/test';

test.beforeEach(async ({ page }) => {
  page.on('pageerror', (error) => {
    throw error;
  });
  await page.goto('/#/diffusion');
  await expect(page.getByRole('heading', { name: /Diffusion models/ })).toBeVisible();
});

test('shows the four sections and the default model', async ({ page }) => {
  for (const title of [
    'Destroying data with noise',
    'What the network learned',
    'Generating: noise to data',
    'Train it',
  ]) {
    await expect(page.getByRole('heading', { name: title })).toBeVisible();
  }
  await expect(page.getByText('50 noising steps')).toBeVisible();
  await expect(page.getByRole('img', { name: /Data points with noise added/ })).toBeVisible();
  await expect(page.getByRole('img', { name: /Linear schedule/i })).toBeVisible();
});

test('the time slider changes how much signal is left', async ({ page }) => {
  const slider = page.getByRole('slider', { name: /Time t/ });
  await slider.focus();
  await page.keyboard.press('Home');
  await expect(page.getByText(/signal kept 100%/)).toBeVisible();
  await page.keyboard.press('End');
  await expect(page.getByText(/signal kept [0-2]%/)).toBeVisible();
});

test('the noising can be played', async ({ page }) => {
  await page.getByRole('slider', { name: /Time t/ }).focus();
  await page.keyboard.press('Home');
  await page.getByRole('button', { name: 'Play the noising' }).click();
  await expect(page.getByText(/signal kept 100%/)).toBeHidden({ timeout: 5000 });
});

test('arrows can be hidden', async ({ page }) => {
  const toggle = page.getByRole('switch', { name: "Show the network's arrows" });
  await expect(toggle).toBeChecked();
  await toggle.click();
  await expect(toggle).not.toBeChecked();
});

test('generation plays through the steps and the sampler can be changed', async ({ page }) => {
  await expect(page.getByText(/0 \/ 20/)).toBeVisible();
  await page.getByRole('button', { name: 'Generate' }).click();
  await expect(page.getByText(/20 \/ 20/)).toBeVisible({ timeout: 10_000 });
  await page.getByRole('radio', { name: 'DDPM' }).click();
  await expect(page.getByText(/DDPM always takes all 50 steps/)).toBeVisible();
  await expect(page.getByText(/0 \/ 50/)).toBeVisible();
});

test('ten steps run exactly ten training iterations', async ({ page }) => {
  await page.getByRole('button', { name: 'Run 10 steps' }).click();
  await expect(page.getByText('10 iterations trained')).toBeVisible();
});

test('changing the number of noising steps restarts the model', async ({ page }) => {
  await page.getByRole('button', { name: 'Run 10 steps' }).click();
  await expect(page.getByText('10 iterations trained')).toBeVisible();
  await page.getByRole('slider', { name: 'Noising steps T' }).focus();
  await page.keyboard.press('ArrowRight');
  await expect(page.getByText('60 noising steps')).toBeVisible();
  await expect(page.getByText('0 iterations trained')).toBeVisible();
});

test('an experiment configures the lab and trains it', async ({ page }) => {
  await page.getByRole('button', { name: 'Learning a curve' }).click();
  await expect(page.getByText(/What to look for/)).toBeVisible();
  await expect(page.getByText('DDPM sampler')).toBeVisible();
  await expect(page.getByRole('button', { name: 'Pause training' })).toBeVisible();
  await expect(page.getByText(/^[1-9]\d* iterations trained$/)).toBeVisible({ timeout: 20_000 });
  await page.getByRole('button', { name: 'Pause training' }).click();
  await page.getByRole('radio', { name: 'DDIM' }).click();
  await expect(page.getByText(/What to look for/)).toBeHidden();
});

test('fits a phone screen without horizontal page scroll', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 800 });
  await page.reload();
  await expect(page.getByRole('heading', { name: /Diffusion models/ })).toBeVisible();
  const overflow = await page.evaluate(
    () => document.documentElement.scrollWidth - document.documentElement.clientWidth,
  );
  expect(overflow).toBeLessThanOrEqual(0);
});
