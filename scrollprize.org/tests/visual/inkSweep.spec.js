// @ts-check
const { test, expect } = require("@playwright/test");

// Minimal smoke test for the ink-render compare sweep. Every PHerc0814
// segment currently has >1 render variant, so the compare toggle must show
// for the first segment's lightbox. A later milestone expands this into full
// regression coverage (drag, keyboard nudge, per-side labels, 3D/2D + segment
// step resets, etc).
test("compare toggle appears in the ink segment lightbox", async ({ page }) => {
  await page.goto("/data_browser/PHerc0814");

  const grid = page.locator(".inkgrid");
  await expect(grid).toBeVisible();

  const firstThumb = grid.locator(".inkcard").first();
  await expect(firstThumb).toBeVisible();
  await firstThumb.locator("a").first().click();

  const lightbox = page.locator(".lightbox");
  await expect(lightbox).toBeVisible();

  await expect(lightbox.locator(".lbcomparetoggle")).toBeVisible();
});
