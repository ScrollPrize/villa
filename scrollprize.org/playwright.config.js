// @ts-check
const { defineConfig, devices } = require("@playwright/test");

// Chromium only: this is a dark-mode-only site, so a single browser engine is
// proportionate for now. NO webServer block — the dev server's lifecycle is
// managed externally (by whatever workflow runs these tests), not by
// Playwright, so we don't re-pay the ~25s S3 metadata fetch on every run.
module.exports = defineConfig({
  testDir: "./tests",
  fullyParallel: true,
  reporter: [["list"]],
  use: {
    baseURL: "http://localhost:3000",
    trace: "on-first-retry",
  },
  projects: [
    {
      name: "chromium",
      use: { ...devices["Desktop Chrome"] },
    },
  ],
});
