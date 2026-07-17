// @ts-check
const { test, expect } = require("@playwright/test");

// Regression suite for the ink-render compare sweep (CompareRenders.js +
// InkSegmentsGallery.js). Grew out of a single smoke test into full coverage:
//   1. PHerc0814 (every segment has 2 render variants): open the lightbox on
//      several segments, toggle compare on, drag the sweep handle across its
//      full range, and confirm zero console/page errors throughout.
//   2. PHerc0172 (nearly every segment has 0/1 variant, i.e. not compare-
//      eligible): confirm the lightbox falls back to the pre-existing
//      single-image view with no compare toggle at all.
//   3. Keyboard regression: Escape closes the lightbox and ←/→ step segments,
//      both with compare off and with focus inside the compare frame (where
//      ←/→ must nudge the divider instead of stepping the segment).
//   4. No `.zoomable` class ever leaks onto the compare frame's <img>s, and
//      dragging the sweep handle never opens the sitewide zoom overlay.
//   5. PHercParis4: the 2D tab shows compare on eligible segments, the 3D tab
//      shows no compare toggle on ANY segment — checked across the whole
//      grid, not just a single sample.

// --- shared helpers -------------------------------------------------------

// Collects console "error" messages and uncaught page errors for the
// lifetime of a page. Assert on the returned array near the end of a test.
function collectConsoleErrors(page) {
  const errors = [];
  page.on("console", (msg) => {
    if (msg.type() === "error") errors.push(`console.error: ${msg.text()}`);
  });
  page.on("pageerror", (err) => {
    errors.push(`pageerror: ${err.message}`);
  });
  return errors;
}

async function openLightboxAt(page, index) {
  const grid = page.locator(".inkgrid");
  await expect(grid).toBeVisible();
  const card = grid.locator(".inkcard").nth(index);
  await expect(card).toBeVisible();
  await card.locator("a").first().click();
  const lightbox = page.locator(".lightbox");
  await expect(lightbox).toBeVisible();
  return lightbox;
}

async function dragHandleToFraction(page, frame, handle, fraction) {
  const frameBox = await frame.boundingBox();
  if (!frameBox) throw new Error("compareframe has no bounding box");
  // Clamp strictly inside the frame's box: boundingBox() width/height are
  // exclusive bounds, and the frame sits centered over the fixed lbprev/
  // lbnext nav buttons + the lightbox backdrop (click-away-to-close) at its
  // very edges — landing exactly on frame.x+width would miss the frame
  // element and hit whatever is layered behind/beside it there instead.
  const x = Math.max(0, Math.min(frameBox.width - 1, frameBox.width * fraction));
  const targetX = frameBox.x + x;
  const targetY = frameBox.y + frameBox.height / 2;
  const handleBox = await handle.boundingBox();
  if (!handleBox) throw new Error("sweep-handle has no bounding box");
  const startX = handleBox.x + handleBox.width / 2;
  const startY = handleBox.y + handleBox.height / 2;
  await page.mouse.move(startX, startY);
  await page.mouse.down();
  await page.mouse.move(targetX, targetY, { steps: 10 });
  await page.mouse.up();
}

// --- 1. PHerc0814: compare drag across several segments, no console errors --

test.describe("PHerc0814 — compare sweep drag", () => {
  test("toggling compare + dragging the handle across 3 segments logs no errors", async ({
    page,
  }) => {
    const errors = collectConsoleErrors(page);
    await page.goto("/data_browser/PHerc0814");

    for (const index of [0, 1, 2]) {
      const lightbox = await openLightboxAt(page, index);

      const toggle = lightbox.locator(".lbcomparetoggle");
      await expect(toggle).toBeVisible();
      await toggle.click();

      const frame = lightbox.locator(".compareframe");
      await expect(frame).toBeVisible();
      const handle = lightbox.locator('[data-testid="sweep-handle"]');
      await expect(handle).toBeVisible();

      // Drag to both extremes and a middle position. Tolerance of a couple
      // percentage points absorbs the sub-pixel clamp dragHandleToFraction
      // applies to stay strictly inside the frame's hit box at the edges.
      for (const fraction of [0, 1, 0.5]) {
        await dragHandleToFraction(page, frame, handle, fraction);
        const expected = fraction * 100;
        await expect
          .poll(async () => Number(await frame.getAttribute("aria-valuenow")))
          .toBeGreaterThanOrEqual(expected - 2);
        await expect
          .poll(async () => Number(await frame.getAttribute("aria-valuenow")))
          .toBeLessThanOrEqual(expected + 2);
      }

      await lightbox.locator(".lbclose").click();
      await expect(lightbox).toBeHidden();
    }

    expect(errors).toEqual([]);
  });
});

// --- 2. Sample with only single-variant segments: old single-image view ---

// A sample's live data (metadata.min.json, preferred over the bundled
// static/data_browser/index.json snapshot — see useAtlasData.js) can gain
// render variants over time, so which segments are compare-eligible drifts.
// Rather than hard-code an index confirmed only against the bundled
// snapshot (fragile against that drift), search the actual rendered grid for
// a segment the LIVE data currently makes non-compare-eligible — as of this
// writing, most non-showcase samples (and most of PHercParis4's 2D segments)
// have exactly this shape.
test.describe("single-variant segment — pre-compare view is unchanged", () => {
  test("a non-compare-eligible segment's lightbox shows the old plain-image view, with no compare toggle", async ({
    page,
  }) => {
    const errors = collectConsoleErrors(page);
    await page.goto("/data_browser/PHercParis4");

    const tablist = page.locator(".segtoggle");
    await expect(tablist).toBeVisible();
    await tablist.getByRole("tab", { name: /2D ink/i }).click();

    const grid = page.locator(".inkgrid");
    await expect(grid).toBeVisible();
    const cards = grid.locator(".inkcard");
    const count = await cards.count();

    let checked = false;
    for (let i = 0; i < count; i++) {
      await cards.nth(i).locator("a").first().click();
      const lightbox = page.locator(".lightbox");
      await expect(lightbox).toBeVisible();

      if ((await lightbox.locator(".lbcomparetoggle").count()) === 0) {
        // Found a non-compare-eligible segment: assert the exact
        // pre-existing single-image view.
        await expect(lightbox.locator(".compareframe")).toHaveCount(0);
        await expect(lightbox.locator(".comparewrap")).toHaveCount(0);

        const img = lightbox.locator("> img");
        await expect(img).toHaveCount(1);
        await expect(img).toHaveAttribute("alt", /^ink prediction — /);
        const src = await img.getAttribute("src");
        expect(src).toMatch(/fit-in\/2000x2000/);

        const caption = lightbox.locator(".lbcap");
        await expect(caption).toContainText(`${i + 1}/${count}`);

        checked = true;
        await lightbox.locator(".lbclose").click();
        await expect(lightbox).toBeHidden();
        break;
      }
      await lightbox.locator(".lbclose").click();
      await expect(lightbox).toBeHidden();
    }

    expect(checked).toBe(true);
    expect(errors).toEqual([]);
  });
});

// --- 3. Keyboard regression: arrow-step / Escape, on + off compare focus ---

test.describe("PHerc0814 — keyboard regression (arrow-step, Escape, compare-focus guard)", () => {
  test("arrows step segments and Escape closes when compare is off", async ({
    page,
  }) => {
    const errors = collectConsoleErrors(page);
    await page.goto("/data_browser/PHerc0814");

    const lightbox = await openLightboxAt(page, 0);
    const caption = lightbox.locator(".lbcap");
    await expect(caption).toContainText("1/19");

    await page.keyboard.press("ArrowRight");
    await expect(caption).toContainText("2/19");

    await page.keyboard.press("ArrowLeft");
    await expect(caption).toContainText("1/19");

    await page.keyboard.press("Escape");
    await expect(lightbox).toBeHidden();

    expect(errors).toEqual([]);
  });

  test("arrows nudge the compare divider (not the segment) when the compare frame has focus, and Escape still closes", async ({
    page,
  }) => {
    const errors = collectConsoleErrors(page);
    await page.goto("/data_browser/PHerc0814");

    const lightbox = await openLightboxAt(page, 0);
    const caption = lightbox.locator(".lbcap");
    await expect(caption).toContainText("1/19");

    await lightbox.locator(".lbcomparetoggle").click();
    const frame = lightbox.locator(".compareframe");
    await expect(frame).toBeVisible();
    await frame.focus();
    await expect(frame).toBeFocused();
    await expect(frame).toHaveAttribute("aria-valuenow", "50");

    // Nudge right: divider moves, segment/caption must NOT change.
    await page.keyboard.press("ArrowRight");
    await expect(frame).toHaveAttribute("aria-valuenow", "52");
    await expect(caption).toContainText("1/19");

    await page.keyboard.press("ArrowLeft");
    await page.keyboard.press("ArrowLeft");
    await expect(frame).toHaveAttribute("aria-valuenow", "48");
    await expect(caption).toContainText("1/19");

    // Escape still closes the lightbox even while the compare frame is
    // focused.
    await page.keyboard.press("Escape");
    await expect(lightbox).toBeHidden();

    expect(errors).toEqual([]);
  });
});

// --- 4. No .zoomable leakage; dragging never opens the sitewide zoom overlay --

test.describe("PHerc0814 — compare frame stays isolated from the sitewide image zoom", () => {
  test("compare frame images are never .zoomable, and dragging never opens the zoom overlay", async ({
    page,
  }) => {
    const errors = collectConsoleErrors(page);
    await page.goto("/data_browser/PHerc0814");

    const lightbox = await openLightboxAt(page, 0);
    await lightbox.locator(".lbcomparetoggle").click();
    const frame = lightbox.locator(".compareframe");
    await expect(frame).toBeVisible();

    await expect(frame.locator("img.zoomable")).toHaveCount(0);
    await expect(frame.locator("img")).toHaveCount(2);

    const handle = lightbox.locator('[data-testid="sweep-handle"]');
    for (const fraction of [0, 0.3, 0.7, 1, 0.5]) {
      await dragHandleToFraction(page, frame, handle, fraction);
    }

    // The sitewide zoom overlay (img-zoom-overlay, imageZoom.js) must never
    // have opened as a side effect of the drag.
    await expect(page.locator(".img-zoom-overlay.is-open")).toHaveCount(0);

    expect(errors).toEqual([]);
  });
});

// --- 5. PHercParis4: 2D compare-eligible, 3D never compare-eligible --------

test.describe("PHercParis4 — 2D/3D toggle compare-eligibility regression", () => {
  test("2D tab: at least one segment offers compare", async ({ page }) => {
    const errors = collectConsoleErrors(page);
    await page.goto("/data_browser/PHercParis4");

    const tablist = page.locator(".segtoggle");
    await expect(tablist).toBeVisible();
    await tablist.getByRole("tab", { name: /2D ink/i }).click();

    const grid = page.locator(".inkgrid");
    await expect(grid).toBeVisible();
    const cards = grid.locator(".inkcard");
    const count = await cards.count();

    let foundCompare = false;
    for (let i = 0; i < count; i++) {
      await cards.nth(i).locator("a").first().click();
      const lightbox = page.locator(".lightbox");
      await expect(lightbox).toBeVisible();
      const toggle = lightbox.locator(".lbcomparetoggle");
      if (await toggle.count()) {
        foundCompare = true;
        await lightbox.locator(".lbclose").click();
        await expect(lightbox).toBeHidden();
        break;
      }
      await lightbox.locator(".lbclose").click();
      await expect(lightbox).toBeHidden();
    }

    expect(foundCompare).toBe(true);
    expect(errors).toEqual([]);
  });

  test("3D tab: NO segment, across the whole grid, ever offers compare", async ({
    page,
  }) => {
    test.setTimeout(120000);
    const errors = collectConsoleErrors(page);
    await page.goto("/data_browser/PHercParis4");

    const tablist = page.locator(".segtoggle");
    await expect(tablist).toBeVisible();
    await tablist.getByRole("tab", { name: /3D ink/i }).click();

    const grid = page.locator(".inkgrid");
    await expect(grid).toBeVisible();
    const cards = grid.locator(".inkcard");
    const count = await cards.count();
    expect(count).toBeGreaterThan(0);

    for (let i = 0; i < count; i++) {
      await cards.nth(i).locator("a").first().click();
      const lightbox = page.locator(".lightbox");
      await expect(lightbox).toBeVisible();
      await expect(lightbox.locator(".lbcomparetoggle")).toHaveCount(0);
      await lightbox.locator(".lbclose").click();
      await expect(lightbox).toBeHidden();
    }

    expect(errors).toEqual([]);
  });
});
