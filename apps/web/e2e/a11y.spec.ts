import { readFileSync } from 'node:fs';
import AxeBuilder from '@axe-core/playwright';
import { expect, test } from '@playwright/test';

// Every route in the registry, checked with axe (WCAG 2.x A and AA) in both themes.
const registry = readFileSync('src/app/registry.ts', 'utf8');
const paths = ['/', ...[...registry.matchAll(/path: '([^']+)'/g)].map((m) => m[1] as string)];

for (const theme of ['light', 'dark'] as const) {
  test.describe(`accessibility (${theme})`, () => {
    test.use({ colorScheme: theme });
    for (const path of paths) {
      test(`${path} has no axe violations`, async ({ page }) => {
        test.setTimeout(180_000);
        await page.addInitScript((t) => localStorage.setItem('sim-theme', t), theme);
        await page.goto(`/#${path}`);
        await page.locator('h1').first().waitFor();
        await page.waitForTimeout(600);
        const { violations } = await new AxeBuilder({ page })
          .withTags(['wcag2a', 'wcag2aa', 'wcag21a', 'wcag21aa'])
          .analyze();
        const summary = violations.map(
          (v) => `${v.id} (${v.impact}) x${v.nodes.length}: ${v.nodes[0]?.target.join(' ')}`,
        );
        expect(summary).toEqual([]);
      });
    }
  });
}
