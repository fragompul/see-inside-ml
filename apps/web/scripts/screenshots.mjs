// Regenerates the README screenshots: `npm run screenshots` (needs the dev server on :5173).
import { mkdir } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { chromium } from '@playwright/test';

const outDir = fileURLToPath(new URL('../../../img/', import.meta.url));
const base = process.env.APP_URL ?? 'http://localhost:5173/';
await mkdir(outDir, { recursive: true });

const browser = await chromium.launch({ channel: process.env.CI ? undefined : 'chrome' });

/** Each shot: where to go, which theme, what to click, and what to capture. */
const shots = [
  { name: 'hero-dark', theme: 'dark', hash: '', settle: 4500 },
  { name: 'hero-light', theme: 'light', hash: '', settle: 4500 },
  { name: 'path-dark', theme: 'dark', hash: '', settle: 4500, only: '#path' },
  {
    name: 'playground-dark',
    theme: 'dark',
    hash: '#/mlp',
    settle: 800,
    steps: async (page) => {
      await page.getByRole('button', { name: 'Spiral challenge' }).click();
      await page.waitForTimeout(7000);
      await page.getByRole('button', { name: 'Pause training' }).click();
      await page.getByRole('button', { name: 'L2 · n3' }).hover();
      await page.waitForTimeout(500);
    },
    height: 1150,
  },
  {
    name: 'playground-light',
    theme: 'light',
    hash: '#/mlp',
    settle: 800,
    steps: async (page) => {
      await page.getByRole('button', { name: 'Feature engineering' }).click();
      await page.waitForTimeout(3500);
      await page.getByRole('button', { name: 'Pause training' }).click();
    },
    height: 1150,
  },
  {
    name: 'inspector-dark',
    theme: 'dark',
    hash: '#/mlp',
    settle: 800,
    steps: async (page) => {
      await page.getByRole('button', { name: 'Step one epoch' }).click();
      await page.waitForTimeout(300);
      await page.getByRole('button', { name: 'Inspect a step' }).click();
      await page.getByRole('button', { name: /Backward/ }).click();
      await page.waitForTimeout(900);
      await page.mouse.move(10, 10);
      await page.getByRole('button', { name: 'L1 · n2' }).hover();
      await page.waitForTimeout(500);
    },
    height: 1400,
  },
  {
    name: 'transformer-dark',
    theme: 'dark',
    hash: '#/transformer',
    settle: 1500,
    steps: async (page) => {
      await page
        .getByRole('button', { name: 'Mia and Leo went to the park. Mia said hello to' })
        .click();
      await page.getByRole('button', { name: 'Generate one token' }).click();
      await page.waitForTimeout(400);
      await page.evaluate(() => document.getElementById('tf-attention')?.scrollIntoView());
      await page.waitForTimeout(700);
    },
    height: 1150,
  },
  {
    name: 'transformer-light',
    theme: 'light',
    hash: '#/transformer',
    settle: 1500,
    steps: async (page) => {
      await page.getByRole('button', { name: /Token 4: / }).click();
      await page.evaluate(() => document.getElementById('tf-embeddings')?.scrollIntoView());
      await page.waitForTimeout(700);
    },
    height: 1150,
  },
  {
    name: 'transformer-3d-dark',
    theme: 'dark',
    hash: '#/transformer',
    settle: 1500,
    steps: async (page) => {
      await page.evaluate(() => document.getElementById('tf-3d')?.scrollIntoView());
      await page.waitForTimeout(2500);
      await page.getByRole('button', { name: /Attention · layer 1/ }).click();
      await page.waitForTimeout(3500);
    },
    height: 1150,
  },
  {
    name: 'cnn-dark',
    theme: 'dark',
    hash: '#/cnn',
    settle: 1200,
    steps: async (page) => {
      await page.getByRole('button', { name: 'Start training' }).click();
      await page.waitForTimeout(9000);
      await page.getByRole('button', { name: 'Pause training' }).click();
      await page.getByRole('switch', { name: 'Show what the decision depends on' }).click();
      await page.evaluate(() => window.scrollTo(0, 150));
      await page.waitForTimeout(600);
    },
    height: 1150,
  },
  {
    name: 'rnn-dark',
    theme: 'dark',
    hash: '#/rnn',
    settle: 1200,
    steps: async (page) => {
      await page.getByRole('button', { name: 'Gates fix it: LSTM' }).click();
      await page.waitForTimeout(9000);
      await page.getByRole('button', { name: 'Pause training' }).click();
      await page.evaluate(() => document.getElementById('rnn-unrolled')?.scrollIntoView());
      await page.waitForTimeout(600);
    },
    height: 1250,
  },
  {
    name: 'autoencoder-dark',
    theme: 'dark',
    hash: '#/autoencoder',
    settle: 1200,
    steps: async (page) => {
      await page.getByRole('button', { name: 'A map you can sample' }).click();
      await page.waitForTimeout(14000);
      await page.getByRole('button', { name: 'Pause training' }).click();
      await page.evaluate(() => document.getElementById('ae-latent')?.scrollIntoView());
      await page.waitForTimeout(600);
    },
    height: 1000,
  },
  {
    name: 'gan-dark',
    theme: 'dark',
    hash: '#/gan',
    settle: 1200,
    steps: async (page) => {
      await page.getByRole('button', { name: 'A fair duel' }).click();
      await page.waitForTimeout(14000);
      await page.getByRole('button', { name: 'Pause training' }).click();
      await page.evaluate(() => window.scrollTo(0, 250));
      await page.waitForTimeout(600);
    },
    height: 1100,
  },
  {
    name: 'diffusion-dark',
    theme: 'dark',
    hash: '#/diffusion',
    settle: 1200,
    steps: async (page) => {
      await page.getByRole('button', { name: 'Skipping steps with DDIM' }).click();
      await page.waitForTimeout(14000);
      await page.getByRole('button', { name: 'Pause training' }).click();
      await page.evaluate(() => document.getElementById('diff-sample')?.scrollIntoView());
      await page.getByRole('button', { name: 'Play generation' }).click();
      await page.waitForTimeout(3500);
    },
    height: 1000,
  },
  {
    name: 'embeddings-dark',
    theme: 'dark',
    hash: '#/nlp',
    settle: 1200,
    steps: async (page) => {
      await page.getByRole('button', { name: 'Capitals are a direction' }).click();
      await page.waitForTimeout(9000);
      await page.getByRole('button', { name: 'Pause training' }).click();
      await page.evaluate(() => document.getElementById('nlp-map')?.scrollIntoView());
      await page.waitForTimeout(600);
    },
    height: 1000,
  },
  {
    name: 'rl-dark',
    theme: 'dark',
    hash: '#/rl',
    settle: 1200,
    steps: async (page) => {
      await page.getByRole('button', { name: 'Q-learning on the cliff' }).click();
      await page.waitForTimeout(5000);
      await page.getByRole('button', { name: 'Pause training' }).click();
      await page.evaluate(() => document.getElementById('rl-world')?.scrollIntoView());
      await page.waitForTimeout(600);
    },
    height: 1000,
  },
  {
    name: 'foundations-dark',
    theme: 'dark',
    hash: '#/foundations',
    settle: 1200,
    steps: async (page) => {
      await page.getByRole('button', { name: 'Momentum' }).click();
      await page.waitForTimeout(4000);
      await page.getByRole('button', { name: 'Pause the ball' }).click();
      await page.evaluate(() => document.getElementById('fnd-descent')?.scrollIntoView());
      await page.waitForTimeout(600);
    },
    height: 1000,
  },
];

for (const shot of shots) {
  const context = await browser.newContext({
    viewport: { width: 1440, height: shot.height ?? 900 },
    deviceScaleFactor: 2,
    reducedMotion: 'no-preference',
  });
  await context.addInitScript((theme) => localStorage.setItem('sim-theme', theme), shot.theme);
  const page = await context.newPage();
  await page.goto(`${base}${shot.hash}`);
  await page.waitForTimeout(shot.settle); // let signals propagate / the app boot
  await shot.steps?.(page);
  const path = `${outDir}${shot.name}.png`;
  if (shot.only) {
    await page.locator(shot.only).scrollIntoViewIfNeeded();
    await page.waitForTimeout(1500);
    await page.locator(shot.only).screenshot({ path });
  } else {
    await page.screenshot({ path });
  }
  await context.close();
}

await browser.close();
console.log(`Screenshots written to ${outDir}`);
