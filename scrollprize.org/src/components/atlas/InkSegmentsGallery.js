import React, { useState, useEffect } from "react";
import { thumbnailUrls, neuroglancerUrl, toHttp } from "./dataAccess";

// InkSegmentsGallery — the per-segment listing the old atlas had: every segment
// with an ink prediction, as a lazy-loaded grid of small Thumbor thumbnails.
//
// Two prediction kinds share these segments (built in scripts/genAtlasData.js
// from metadata.json):
//   • 2D ink (flattened): `down` (downsampled .jpg) / `full` (.tif).
//   • 3D ink (on surface): `alphaPrev` (ds8 .jpg) / `alpha` (.tif).
// When a sample has both (PHercParis4), we show a 3D/2D toggle over ONE grid.
// Clicking a thumbnail opens an in-page lightbox (a Thumbor-resized view of the
// .jpg — the raw downsampled files can be tens of MB); ← / → (or the ‹ ›
// buttons) step through the segments. Where available each card also links its
// surface layers in Neuroglancer and its mesh.

export default function InkSegmentsGallery({ segments, display }) {
  const all = segments || [];
  const n2D = all.filter((s) => s.full).length;
  const n3D = all.filter((s) => s.alphaPrev).length;
  // Default to whichever view has more entries (3D on PHercParis4: 43 vs 41).
  const [view, setView] = useState(n3D >= n2D && n3D > 0 ? "3d" : "2d");
  // Lightbox: index into `shown` of the image in the popup, or null.
  const [zoomIdx, setZoomIdx] = useState(null);
  const [imgLoading, setImgLoading] = useState(false);

  // Active view + the segments it shows (also drives lightbox navigation).
  const v = view === "3d" && n3D > 0 ? "3d" : n2D > 0 ? "2d" : "3d";
  const is3d = v === "3d";
  const shown = all.filter((s) => (is3d ? s.alphaPrev : s.full));
  const n = shown.length;

  // Lightbox keyboard: Esc closes; ← / → step through `shown` (wrapping). Body
  // scroll is locked while open. Client-only — window/document untouched in SSR.
  useEffect(() => {
    if (zoomIdx == null) return undefined;
    const onKey = (e) => {
      if (e.key === "Escape") setZoomIdx(null);
      else if (e.key === "ArrowLeft") {
        e.preventDefault();
        setZoomIdx((i) => (i == null ? i : (i + n - 1) % n));
      } else if (e.key === "ArrowRight") {
        e.preventDefault();
        setZoomIdx((i) => (i == null ? i : (i + 1) % n));
      }
    };
    window.addEventListener("keydown", onKey);
    const prevOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => {
      window.removeEventListener("keydown", onKey);
      document.body.style.overflow = prevOverflow;
    };
  }, [zoomIdx, n]);

  // Show the spinner each time the popup opens or steps to a new image; the
  // <img> onLoad/onError clears it.
  useEffect(() => {
    if (zoomIdx != null) setImgLoading(true);
  }, [zoomIdx]);

  if (!n2D && !n3D) return null;

  const showToggle = n2D > 0 && n3D > 0;

  // Current lightbox image, derived from the live `shown` list + index.
  const cur = zoomIdx != null ? shown[zoomIdx] : null;
  const lb = cur
    ? {
        jpg: is3d ? cur.alphaPrev : cur.down,
        full: is3d ? cur.alpha : cur.full,
        name: cur.label || cur.id,
        label: `${cur.label || cur.id}${cur.um ? ` · ${cur.um}µm` : ""}`,
      }
    : null;

  return (
    <div className="panel full inkseg">
      <h2>
        Ink predictions ({n} segment{n === 1 ? "" : "s"})
      </h2>
      <p className="txt" style={{ marginBottom: showToggle ? "10px" : "12px" }}>
        {is3d
          ? "Ink detected and painted back onto the rendered 3D surface of each segment."
          : "Ink detected on each flattened segment surface."}{" "}
        Click a thumbnail to view it (← / → to step through); each segment also
        links to the full-resolution TIFF, and where available its surface layers
        in Neuroglancer and its mesh.
      </p>

      {showToggle ? (
        <div className="segtoggle" role="tablist" aria-label="prediction view">
          <button
            type="button"
            role="tab"
            aria-selected={is3d}
            className={is3d ? "on" : ""}
            onClick={() => setView("3d")}
          >
            3D ink ({n3D})
          </button>
          <button
            type="button"
            role="tab"
            aria-selected={!is3d}
            className={!is3d ? "on" : ""}
            onClick={() => setView("2d")}
          >
            2D ink ({n2D})
          </button>
        </div>
      ) : null}

      <div className="inkgrid">
        {shown.map((s, i) => {
          const previewUrl = is3d ? s.alphaPrev : s.preview;
          const jpgUrl = is3d ? s.alphaPrev : s.down; // downsampled JPG
          const fullUrl = is3d ? s.alpha : s.full; // full-resolution TIFF
          const { serviceUrl } = thumbnailUrls(previewUrl, 400, 400);
          const name = s.label || s.id;
          const label = `${name}${s.um ? ` · ${s.um}µm` : ""}`;
          const ng = s.layers ? neuroglancerUrl(s.layers, name) : null;
          return (
            <div key={s.id || i} className="inkcard">
              <a
                href={jpgUrl || fullUrl}
                target="_blank"
                rel="noopener noreferrer"
                title={is3d ? "view 3D-ink render" : "view ink prediction"}
                onClick={(e) => {
                  if (jpgUrl) {
                    e.preventDefault();
                    setZoomIdx(i);
                  }
                }}
              >
                <img
                  src={serviceUrl || previewUrl}
                  alt={`${display || ""} ${
                    is3d ? "3D-ink render" : "ink prediction"
                  } — segment ${name}`}
                  loading="lazy"
                  onError={(e) => {
                    e.currentTarget.style.display = "none";
                  }}
                />
              </a>
              <span className="inklabel">{label}</span>
              <span className="inklinks">
                {jpgUrl ? (
                  <a
                    href={jpgUrl}
                    target="_blank"
                    rel="noopener noreferrer"
                    title="downsampled JPG — opens in the browser"
                  >
                    jpg ↗
                  </a>
                ) : null}
                {fullUrl ? (
                  <a
                    href={fullUrl}
                    target="_blank"
                    rel="noopener noreferrer"
                    title="full-resolution TIFF — downloads"
                  >
                    tif ↗
                  </a>
                ) : null}
                {ng ? (
                  <a href={ng} target="_blank" rel="noopener noreferrer">
                    3D ↗
                  </a>
                ) : null}
                {s.mesh ? (
                  <a href={toHttp(s.mesh)} target="_blank" rel="noopener noreferrer">
                    mesh ↗
                  </a>
                ) : null}
              </span>
            </div>
          );
        })}
      </div>

      {lb ? (
        <div
          className="lightbox"
          role="dialog"
          aria-modal="true"
          onClick={(e) => {
            if (e.target === e.currentTarget) setZoomIdx(null);
          }}
        >
          <button
            type="button"
            className="lbclose"
            aria-label="Close"
            onClick={() => setZoomIdx(null)}
          >
            ×
          </button>
          {n > 1 ? (
            <>
              <button
                type="button"
                className="lbprev"
                aria-label="Previous segment"
                onClick={() => setZoomIdx((i) => (i + n - 1) % n)}
              >
                ‹
              </button>
              <button
                type="button"
                className="lbnext"
                aria-label="Next segment"
                onClick={() => setZoomIdx((i) => (i + 1) % n)}
              >
                ›
              </button>
            </>
          ) : null}
          {imgLoading ? <div className="lbspin" aria-label="loading" /> : null}
          <img
            src={thumbnailUrls(lb.jpg, 2000, 2000).serviceUrl || lb.jpg}
            alt={`ink prediction — ${lb.name}`}
            onLoad={() => setImgLoading(false)}
            onError={() => setImgLoading(false)}
          />
          <div className="lbcap">
            {lb.label}
            {n > 1 ? ` · ${zoomIdx + 1}/${n}` : ""}
            {lb.full ? (
              <>
                {" · "}
                <a href={lb.full} target="_blank" rel="noopener noreferrer">
                  full TIFF ↗
                </a>
              </>
            ) : null}
          </div>
        </div>
      ) : null}
    </div>
  );
}
