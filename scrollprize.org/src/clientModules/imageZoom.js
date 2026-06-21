/**
 * Lightweight click-to-zoom lightbox for content images — no dependencies.
 *
 * Clicking any image marked `.zoomable` (added by the rehype image-dimensions
 * plugin, and on selected homepage images) opens an overlay showing the
 * full-resolution image (`data-zoom-src` if present, otherwise the image's own
 * source). Click anywhere or press Escape to close. A single delegated listener
 * survives Docusaurus client-side route changes.
 */
if (typeof document !== "undefined") {
  let overlay = null;

  function ensureOverlay() {
    if (overlay) return overlay;
    overlay = document.createElement("div");
    overlay.className = "img-zoom-overlay";
    overlay.setAttribute("role", "dialog");
    overlay.setAttribute("aria-modal", "true");
    const img = document.createElement("img");
    img.className = "img-zoom-overlay__img";
    img.alt = "";
    overlay.appendChild(img);
    overlay.addEventListener("click", close);
    document.body.appendChild(overlay);
    return overlay;
  }

  function open(src, alt) {
    const o = ensureOverlay();
    const img = o.querySelector(".img-zoom-overlay__img");
    img.src = src;
    img.alt = alt || "";
    o.classList.add("is-open");
    document.documentElement.style.overflow = "hidden";
  }

  function close() {
    if (!overlay) return;
    overlay.classList.remove("is-open");
    document.documentElement.style.overflow = "";
  }

  document.addEventListener("click", (e) => {
    const t = e.target;
    if (!(t instanceof HTMLImageElement)) return;
    if (!t.classList.contains("zoomable")) return;
    // ignore tiny/icon images
    if (t.naturalWidth && t.naturalWidth < 80) return;
    const full = t.getAttribute("data-zoom-src") || t.currentSrc || t.src;
    if (!full) return;
    open(full, t.alt);
  });

  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") close();
  });
}
