import { defineConfig } from 'vitest/config';

export default defineConfig({
  test: {
    // Without this, vitest's default include glob also picks up
    // dist/index.test.js (compiled output from `npm run build`) as a
    // second, duplicate test file alongside src/index.test.ts - found by
    // running the full suite after a build and seeing 44 tests instead
    // of the expected 22.
    exclude: ['dist/**', 'node_modules/**'],
  },
});
