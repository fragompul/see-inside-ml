import { expect, test } from '@playwright/test';

test.beforeEach(async ({ page }) => {
  page.on('pageerror', (error) => {
    throw error;
  });
  await page.goto('/#/autoencoder');
  await expect(page.getByRole('heading', { name: /Autoencoders/ })).toBeVisible();
});

test('shows the four sections and the default model', async ({ page }) => {
  for (const title of [
    'Squeeze a picture, rebuild it',
    'The latent space',
    'Interpolation and imagination',
    'Train it',
  ]) {
    await expect(page.getByRole('heading', { name: title })).toBeVisible();
  }
  await expect(page.getByText('2 numbers in the code')).toBeVisible();
  await expect(page.getByRole('list', { name: 'Code of the picture' })).toBeVisible();
  await expect(page.getByRole('img', { name: /Map of the latent space/ })).toBeVisible();
});

test('picking a digit shows it, encodes it and clearing removes the code', async ({ page }) => {
  await page.getByRole('button', { name: 'Show a 7' }).click();
  await expect(page.getByRole('button', { name: 'Show a 7' })).toHaveAttribute(
    'aria-pressed',
    'true',
  );
  const code = page.getByRole('list', { name: 'Code of the picture' });
  await expect(code.getByText('–')).toHaveCount(0);
  await page.getByRole('button', { name: 'Clear' }).click();
  await expect(code.getByText('–')).toHaveCount(2);
});

test('drawing on the pad produces a reconstruction', async ({ page }) => {
  const pad = page.getByRole('img', { name: /Drawing pad/ });
  const box = (await pad.boundingBox())!;
  await page.mouse.move(box.x + box.width * 0.5, box.y + box.height * 0.2);
  await page.mouse.down();
  await page.mouse.move(box.x + box.width * 0.5, box.y + box.height * 0.8, { steps: 8 });
  await page.mouse.up();
  await expect(page.getByText('Draw a digit here')).toBeHidden();
  await expect(page.getByText(/^Error/)).toContainText(/\d\.\d{3}/);
});

test('noise on the input is shown to the learner', async ({ page }) => {
  await expect(page.getByText('What the encoder receives')).toBeHidden();
  await page.getByRole('slider', { name: 'Noise added to the input' }).focus();
  await page.keyboard.press('ArrowRight');
  await page.keyboard.press('ArrowRight');
  await expect(page.getByText('What the encoder receives')).toBeVisible();
});

test('clicking the latent map moves the probe', async ({ page }) => {
  const slider = page.getByRole('slider', { name: /Probe z1/ });
  await expect(slider).toHaveAttribute('aria-valuenow', '0');
  const map = page.getByRole('img', { name: /Map of the latent space/ });
  await map.scrollIntoViewIfNeeded();
  const box = (await map.boundingBox())!;
  await page.mouse.click(box.x + box.width * 0.85, box.y + box.height * 0.5);
  await expect
    .poll(async () => Number(await slider.getAttribute('aria-valuenow')))
    .toBeGreaterThan(0.5);
});

test('switching to a VAE reveals β and the prior circles', async ({ page }) => {
  await expect(page.getByText(/β \(weight of the KL term\)/)).toBeHidden();
  await page.getByRole('radio', { name: 'Variational (VAE)' }).click();
  await expect(page.getByText(/β \(weight of the KL term\)/)).toBeVisible();
  await expect(page.getByText(/Dashed circles show 1 and 2 standard deviations/)).toBeVisible();
  await expect(page.getByText('(mean ± spread)')).toBeVisible();
});

test('a bigger code adds latent axes to choose from', async ({ page }) => {
  await expect(page.getByLabel('Horizontal axis')).toHaveCount(0);
  await page.getByRole('slider', { name: 'Numbers in the code' }).focus();
  await page.keyboard.press('ArrowRight');
  await expect(page.getByText('3 numbers in the code')).toBeVisible();
  await expect(page.getByLabel('Horizontal axis')).toBeVisible();
  await expect(page.getByRole('slider', { name: /Probe z3/ })).toBeVisible();
});

test('interpolation endpoints and imagination can be changed', async ({ page }) => {
  await page.getByRole('button', { name: 'From a 1' }).click();
  await expect(page.getByRole('button', { name: 'From a 1' })).toHaveAttribute(
    'aria-pressed',
    'true',
  );
  await expect(page.getByRole('list', { name: 'Walking through the code' })).toBeVisible();
  await page.getByRole('button', { name: 'Draw new codes' }).click();
  await expect(page.getByRole('list', { name: 'Images imagined from random codes' })).toBeVisible();
});

test('an experiment configures the lab and trains it', async ({ page }) => {
  await page.getByRole('button', { name: 'A map you can sample' }).click();
  await expect(page.getByText(/What to look for/)).toBeVisible();
  await expect(page.getByText('Variational autoencoder', { exact: true })).toBeVisible();
  await expect(page.getByRole('button', { name: 'Pause training' })).toBeVisible();
  await expect(page.getByText(/test reconstruction/)).toBeVisible({ timeout: 20_000 });
  await page.getByRole('button', { name: 'Pause training' }).click();
  await page.getByRole('radio', { name: 'Autoencoder', exact: true }).click();
  await expect(page.getByText(/What to look for/)).toBeHidden();
});

test('fits a phone screen without horizontal page scroll', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 800 });
  await page.reload();
  await expect(page.getByRole('heading', { name: /Autoencoders/ })).toBeVisible();
  const overflow = await page.evaluate(
    () => document.documentElement.scrollWidth - document.documentElement.clientWidth,
  );
  expect(overflow).toBeLessThanOrEqual(0);
});
