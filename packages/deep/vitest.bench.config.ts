import { defineConfig } from 'vitest/config';

// Benchmarks are opt-in (`npm run bench -w @nnp/engine`) and never run with the unit tests.
export default defineConfig({ test: { include: ['bench/**/*.bench.ts'], testTimeout: 120_000 } });
