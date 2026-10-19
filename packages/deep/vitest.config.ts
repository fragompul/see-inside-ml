import { defineConfig } from 'vitest/config';

// Several suites train small networks; under a busy CPU (CI, many workers) the 5 s default is tight.
export default defineConfig({
  test: { testTimeout: 60_000 },
});
