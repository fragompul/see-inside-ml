// Size budget for the production build. Run after `npm run build`.
// The first load is the entry chunk plus the shared vendor chunks; lab pages and the 3D scene are lazy.
import { readdirSync, readFileSync } from 'node:fs';
import { gzipSync } from 'node:zlib';
import { join } from 'node:path';

const dir = join('apps', 'web', 'dist', 'assets');
const gz = (f) => gzipSync(readFileSync(join(dir, f))).length;
const files = readdirSync(dir).filter((f) => f.endsWith('.js'));
const budgets = [
  {
    name: 'entry + shared (first load)',
    match: /^(index|design-system|lab-kit|src|katex)-/,
    max: 480_000,
  },
  { name: 'largest single chunk (3D scene)', match: /^Scene-/, max: 300_000 },
];
let failed = false;
for (const b of budgets) {
  const total = files.filter((f) => b.match.test(f)).reduce((n, f) => n + gz(f), 0);
  const ok = total <= b.max;
  failed ||= !ok;
  console.log(
    `${ok ? 'ok  ' : 'FAIL'} ${b.name}: ${(total / 1024).toFixed(0)} kB gzip (budget ${(b.max / 1024).toFixed(0)} kB)`,
  );
}
process.exit(failed ? 1 : 0);
