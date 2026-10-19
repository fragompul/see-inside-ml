import { expect, test, type Page } from '@playwright/test';

const tokenCount = async (page: Page) => {
  const text = await page.getByText(/The model sees \d+ tokens/).textContent();
  return Number(/(\d+)/.exec(text ?? '')?.[1]);
};

test.beforeEach(async ({ page }) => {
  page.on('pageerror', (error) => {
    throw error;
  });
  await page.goto('/#/transformer');
  await expect(page.getByRole('heading', { name: /Transformer explorer/ })).toBeVisible();
  await expect(page.getByRole('list', { name: 'Tokens', exact: true })).toBeVisible();
});

test('loads the model and shows its anatomy', async ({ page }) => {
  await expect(page.getByText(/parameters/i).first()).toBeVisible();
  await expect(page.getByText('2 layers')).toBeVisible();
  await expect(page.getByText('4 heads')).toBeVisible();
  await expect(page.getByRole('button', { name: /Token 1: Anna/ })).toBeVisible();
  for (const title of [
    'Prompt and tokens',
    'Embeddings: from ids to vectors',
    'Attention: who reads whom',
    'The residual stream and the MLP',
    'Choosing the next token',
  ]) {
    await expect(page.getByRole('heading', { name: title })).toBeVisible();
  }
});

test('editing the prompt re-tokenizes it', async ({ page }) => {
  const before = await tokenCount(page);
  await page.getByLabel(/Write something/).fill('Tom found a key. He took it to the park.');
  await expect.poll(() => tokenCount(page)).toBeGreaterThan(before);
  await expect(page.getByRole('button', { name: /Token 1: Tom/ })).toBeVisible();

  await page.getByRole('button', { name: 'The small mice play in the forest. They' }).click();
  await expect(page.getByRole('button', { name: /Token 1: The/ })).toBeVisible();
});

test('generates one token at a time, deterministically for the seed', async ({ page }) => {
  const before = await tokenCount(page);
  await page.getByRole('button', { name: 'Generate one token' }).click();
  await expect.poll(() => tokenCount(page)).toBe(before + 1);
  await page.getByRole('button', { name: 'Generate one token' }).click();
  await expect.poll(() => tokenCount(page)).toBe(before + 2);

  await page.getByRole('button', { name: 'Clear generated text' }).click();
  await expect.poll(() => tokenCount(page)).toBe(before);
});

test('auto-generate writes a continuation and can be paused', async ({ page }) => {
  const before = await tokenCount(page);
  await page.getByRole('button', { name: 'Generate text' }).click();
  await expect.poll(() => tokenCount(page)).toBeGreaterThan(before + 2);
  await page.getByRole('button', { name: 'Pause generation' }).click();
  // wait for the paused state to be rendered before sampling the counter
  await expect(page.getByRole('button', { name: 'Generate text' })).toBeVisible();
  const paused = await tokenCount(page);
  await page.waitForTimeout(500);
  expect(await tokenCount(page)).toBe(paused);
});

test('clicking a probability row lets you choose the next token yourself', async ({ page }) => {
  const before = await tokenCount(page);
  const firstRow = page
    .getByRole('list', { name: 'Most likely next tokens' })
    .getByRole('button')
    .first();
  await firstRow.click();
  await expect.poll(() => tokenCount(page)).toBe(before + 1);
});

test('the sampling controls reshape the distribution', async ({ page }) => {
  const top = page
    .getByRole('list', { name: 'Most likely next tokens' })
    .getByRole('button')
    .first();
  const percent = async () => Number((await top.innerText()).match(/(\d+(?:\.\d+)?)%/)?.[1]);

  const base = await percent();
  await page.getByRole('slider', { name: 'Temperature' }).focus();
  await page.keyboard.press('Home'); // temperature 0: greedy, the top token takes everything
  await expect.poll(percent).toBeGreaterThan(base);
  await expect.poll(percent).toBeGreaterThan(99);

  await page.getByRole('slider', { name: 'Temperature' }).press('End'); // 2.0: flatter
  await expect.poll(percent).toBeLessThan(base);

  await page.getByRole('slider', { name: /Top-k/ }).press('Home');
  await page.getByRole('slider', { name: /Top-k/ }).press('ArrowRight');
  await expect(page.getByText(/1 of 512 tokens kept/)).toBeVisible();
});

test('selecting a token focuses every explanation on it', async ({ page }) => {
  await page.getByRole('button', { name: /Token 2: / }).click();
  await expect(page.getByText(/after token 2/)).toBeVisible();
  await expect(page.getByText(/Token 2:/).first()).toBeVisible();
  await page.getByRole('button', { name: /Token 2: / }).click(); // click again to release
  await expect(page.getByText(/after token 2/)).toHaveCount(0);
});

test('the attention explorer switches layer and head and reports the maths', async ({ page }) => {
  await page.getByRole('radio', { name: 'Layer 2' }).first().click();
  await page.getByRole('button', { name: /Head 3/ }).click();
  await expect(page.getByText('Layer 2 · head 3')).toBeVisible();

  const matrix = page.getByRole('grid', { name: /Attention weights/ });
  await matrix.scrollIntoViewIfNeeded();
  const box = await matrix.boundingBox();
  if (!box) throw new Error('no attention matrix');
  await page.mouse.move(box.x + 6, box.y + box.height - 6);
  await expect(page.getByText('Cell under the cursor')).toBeVisible();
  await expect(page.getByText(/softmax/).first()).toBeVisible();
});

test('the tokenizer demo shows BPE merging an unseen word', async ({ page }) => {
  await page.getByLabel(/Try a word/).fill('Zebra');
  await expect(page.getByText(/Start: \d+ raw bytes/)).toBeVisible();
  await page.getByRole('button', { name: 'Next merge' }).click();
  await expect(page.getByText(/Merge 1:/)).toBeVisible();
  await page.getByLabel(/Try a word/).fill('cat');
  await expect(page.getByText(/Start: \d+ raw bytes/)).toBeVisible();
});

test('hovering the embedding map reveals nearest neighbours', async ({ page }) => {
  const svg = page.getByRole('img', { name: /Map of the token embeddings/ });
  await expect(svg).toBeVisible();
  await svg.locator('circle').nth(30).hover({ force: true });
  await expect(page.getByText(/Closest to/)).toBeVisible();
});

test('fits a phone screen without horizontal page scroll', async ({ page }) => {
  await page.setViewportSize({ width: 375, height: 812 });
  await page.goto('/#/transformer');
  await expect(page.getByRole('list', { name: 'Tokens', exact: true })).toBeVisible();
  const overflow = await page.evaluate(
    () => document.documentElement.scrollWidth - document.documentElement.clientWidth,
  );
  expect(overflow).toBeLessThanOrEqual(0);
});

test('experiments load a prompt, a focus and an explanation', async ({ page }) => {
  await page.getByRole('button', { name: /Anna → “She”/ }).click();
  await expect(page.getByLabel(/Write something/)).toHaveValue('Anna found a ball.');
  await expect(page.getByText(/What to look for/)).toBeVisible();
  const top = page
    .getByRole('list', { name: 'Most likely next tokens' })
    .getByRole('button')
    .first();
  await expect(top).toContainText('She');

  await page.getByRole('button', { name: /Follow the attention/ }).click();
  await expect(page.getByText('Layer 2 · head 1')).toBeVisible();
  await expect(page.getByRole('button', { name: /Head 1/ })).toHaveAttribute(
    'aria-pressed',
    'true',
  );

  await page.getByRole('button', { name: 'Temperature 0' }).click();
  await expect(page.getByText('0 · greedy')).toBeVisible();
});

test('editing anything releases the experiment', async ({ page }) => {
  await page.getByRole('button', { name: /Tom → “He”/ }).click();
  await expect(page.getByText(/What to look for/)).toBeVisible();
  await page.getByLabel(/Write something/).fill('Tom found a key.');
  await expect(page.getByText(/What to look for/)).toHaveCount(0);
});

test.describe('3D view', () => {
  test.beforeEach(async ({ page }) => {
    await page.locator('#tf-3d').scrollIntoViewIfNeeded();
    await expect(page.getByTestId('three-host').locator('canvas')).toBeVisible();
  });

  test('renders a WebGL scene, not an empty canvas', async ({ page }) => {
    const canvas = page.getByTestId('three-host').locator('canvas');
    await page.waitForTimeout(800); // let the first frames and the camera fly-in settle
    const png = await canvas.screenshot();
    // an empty transparent canvas compresses to a few hundred bytes
    expect(png.length).toBeGreaterThan(8_000);
  });

  test('the guided tour moves between stages and narrates them', async ({ page }) => {
    const host = page.getByTestId('three-host');
    await expect(host).toHaveAttribute('data-stage', 'overview');

    await page.getByRole('button', { name: /Attention · layer 1/ }).click();
    await expect(host).toHaveAttribute('data-stage', 'attention-0');
    await expect(page.getByRole('heading', { name: /2\. Attention · layer 1/ })).toBeVisible();

    await page.getByRole('button', { name: 'Next stage' }).click();
    await expect(host).toHaveAttribute('data-stage', 'add-0');

    await page.getByRole('button', { name: /Next-token probabilities/ }).click();
    await expect(host).toHaveAttribute('data-stage', 'output');
    await expect(page.getByRole('button', { name: 'Next stage' })).toBeDisabled();

    await page.getByRole('button', { name: 'Overview' }).click();
    await expect(host).toHaveAttribute('data-stage', 'overview');
  });

  test('hovering a cell explains it', async ({ page }) => {
    const canvas = page.getByTestId('three-host').locator('canvas');
    await page
      .getByRole('button', { name: /Embedding/ })
      .first()
      .click();
    await page.waitForTimeout(2200); // camera flight
    const box = await canvas.boundingBox();
    if (!box) throw new Error('no canvas');
    // sweep the middle of the view until a slab cell is under the pointer
    let found = false;
    for (let i = 0; i < 25 && !found; i++) {
      await page.mouse.move(box.x + box.width * (0.35 + 0.012 * i), box.y + box.height * 0.5);
      found = await page.getByText(/dimension \d+/).isVisible();
    }
    expect(found).toBe(true);
    await expect(page.getByText(/value -?\d/)).toBeVisible();
  });
});
