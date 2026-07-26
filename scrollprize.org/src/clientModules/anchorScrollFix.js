/**
 * anchorScrollFix — make #hash links reliably land on their target heading.
 *
 * The problem: these docs pages load images and other media lazily. Docusaurus
 * (and the browser, on a direct URL load) scrolls to the #hash target as soon
 * as the route renders — but media ABOVE the target keeps loading afterward,
 * growing the page and pushing the target down. The reader ends up parked
 * hundreds of pixels above the heading they aimed at. It affects cross-page
 * links, direct URL loads, and same-page table-of-contents clicks alike, and
 * `scroll-behavior: smooth` makes it worse by animating toward a stale spot.
 *
 * The fix: after any navigation that carries a hash, keep the target pinned to
 * its correct offset. A ResizeObserver watches the document height and re-pins
 * the moment anything above the target loads and shifts it — so late-loading
 * images are caught, not just the ones present at first paint. We stop when the
 * reader takes over (scroll wheel / touch / a scroll key) or after a short
 * safety window. scrollIntoView honors each heading's CSS scroll-margin-top, so
 * the landing offset matches native anchor behavior.
 */

let teardown = null;

function pinHash(rawHash) {
  // Cancel any correction still running from a previous navigation.
  if (teardown) {
    teardown();
    teardown = null;
  }

  if (typeof window === "undefined" || typeof document === "undefined") return;
  if (!rawHash || rawHash === "#") return;

  let id;
  try {
    id = decodeURIComponent(rawHash.slice(1));
  } catch (e) {
    id = rawHash.slice(1);
  }
  if (!id) return;

  let stopped = false;
  let ro = null;
  let safetyTimer = null;

  const stop = () => {
    stopped = true;
    if (ro) {
      ro.disconnect();
      ro = null;
    }
    if (safetyTimer) {
      clearTimeout(safetyTimer);
      safetyTimer = null;
    }
    window.removeEventListener("wheel", onUser);
    window.removeEventListener("touchmove", onUser);
    window.removeEventListener("keydown", onKey);
    if (teardown === stop) teardown = null;
  };
  teardown = stop;

  const pin = () => {
    if (stopped) return;
    const el = document.getElementById(id);
    if (!el) return;
    // "instant" so we don't ride the global scroll-behavior: smooth and animate
    // on every re-pin; block:"start" + the element's own scroll-margin-top give
    // the same offset as a native anchor jump.
    el.scrollIntoView({ behavior: "instant", block: "start" });
  };

  const onUser = () => stop();
  const onKey = (e) => {
    // Any deliberate scrolling key hands control back to the reader.
    if (
      e.key === "ArrowUp" ||
      e.key === "ArrowDown" ||
      e.key === "PageUp" ||
      e.key === "PageDown" ||
      e.key === "Home" ||
      e.key === "End" ||
      e.key === " " ||
      e.key === "Spacebar"
    ) {
      stop();
    }
  };

  // Genuine user intent only — our own programmatic scrollIntoView does not
  // emit wheel/touch/key events, so this never self-cancels.
  window.addEventListener("wheel", onUser, { passive: true });
  window.addEventListener("touchmove", onUser, { passive: true });
  window.addEventListener("keydown", onKey);

  // Re-pin whenever the page height changes — i.e. exactly when lazy media above
  // the target finishes loading and would otherwise shove the target down.
  if (typeof ResizeObserver !== "undefined") {
    ro = new ResizeObserver(() => pin());
    ro.observe(document.body);
  }

  // Immediate pins: after Docusaurus' own scroll (setTimeout 0), and across the
  // next couple of frames in case the target element renders a tick late.
  setTimeout(pin, 0);
  requestAnimationFrame(() => {
    pin();
    requestAnimationFrame(pin);
  });

  // Safety net: stop after a few seconds even if something keeps resizing (a
  // looping video, say) so we never pin indefinitely.
  safetyTimer = setTimeout(stop, 4000);
}

export function onRouteDidUpdate({ location }) {
  if (location && location.hash) pinHash(location.hash);
}
