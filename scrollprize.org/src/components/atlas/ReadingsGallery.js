import React, { useState, useEffect } from "react";

// ReadingsGallery — the "Ink detection & renders" panel.
// Ported from the reference renderer (ref/scroll.html ~142-152). Unlike the
// reference, embargoed readings are stripped to null upstream, so anything we
// receive here is PUBLIC: we deliberately render NO embargo banner. Each image
// `src` is already an absolute `/img/data_browser/readings/...` path.
//
// These curated images are modestly sized, so clicking one opens a lightweight,
// dependency-free lightbox (Esc / click / × to close). The per-segment ink and
// 3D-ink galleries deliberately do NOT use this — their full images are very
// large .tif files and stay as new-tab link-outs.

export default function ReadingsGallery({ readings, display }) {
  const [zoom, setZoom] = useState(null); // src of the zoomed image, or null

  // Close on Escape and lock body scroll while the lightbox is open. The effect
  // is client-only, so document/window are never touched during SSR.
  useEffect(() => {
    if (!zoom) return undefined;
    const onKey = (e) => {
      if (e.key === "Escape") setZoom(null);
    };
    window.addEventListener("keydown", onKey);
    const prevOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => {
      window.removeEventListener("keydown", onKey);
      document.body.style.overflow = prevOverflow;
    };
  }, [zoom]);

  if (!readings) return null;

  const images = readings.images || [];
  const n = images.length;

  return (
    <div className="panel full readings">
      <h2>
        Ink detection &amp; renders ({n} image{n === 1 ? "" : "s"})
      </h2>
      {readings.status ? (
        <p className="txt" style={{ marginBottom: "12px" }}>
          {readings.status}
        </p>
      ) : null}
      {images.map((im, i) => {
        const cap = im.cap || "";
        const alt = display ? `${display}${cap ? ` — ${cap}` : ""}` : cap;
        return (
          <React.Fragment key={im.src || i}>
            <img
              src={im.src}
              alt={alt}
              loading="lazy"
              onClick={() => setZoom(im.src)}
            />
            <div className="cap">
              {cap ? `${cap} · ` : ""}
              <a href={im.src} target="_blank" rel="noopener noreferrer">
                full image ↗
              </a>
            </div>
          </React.Fragment>
        );
      })}

      {zoom ? (
        <div
          className="lightbox"
          role="dialog"
          aria-modal="true"
          onClick={() => setZoom(null)}
        >
          <button
            type="button"
            className="lbclose"
            aria-label="Close"
            onClick={() => setZoom(null)}
          >
            ×
          </button>
          <img src={zoom} alt={display ? `${display} — full image` : "full image"} />
        </div>
      ) : null}
    </div>
  );
}
