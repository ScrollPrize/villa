import React, { useState } from "react";
import { thumbnailUrls, neuroglancerUrl, toHttp } from "./dataAccess";

// InkSegmentsGallery — the per-segment listing the old atlas had: every segment
// with an ink prediction, as a lazy-loaded grid of small Thumbor thumbnails.
//
// Two prediction kinds share these segments (built in scripts/genAtlasData.js
// from metadata.json):
//   • 2D ink (flattened): `preview` (downsampled) thumbnail → `full` image.
//   • 3D ink (on surface): `alphaPrev` (ds8 .jpg) thumbnail → `alpha` (.tif).
// When a sample has both (PHercParis4), we show a 3D/2D toggle over ONE grid
// rather than two galleries. Where available each card also links its surface
// layers in Neuroglancer and its mesh. Captions are segment id + resolution
// only — no internal model codenames.

export default function InkSegmentsGallery({ segments, display }) {
  const all = segments || [];
  const n2D = all.filter((s) => s.full).length;
  const n3D = all.filter((s) => s.alphaPrev).length;
  // Default to whichever view has more entries (3D on PHercParis4: 43 vs 41).
  const [view, setView] = useState(n3D >= n2D && n3D > 0 ? "3d" : "2d");

  if (!n2D && !n3D) return null;

  const showToggle = n2D > 0 && n3D > 0;
  // Guard against a stale view if only one kind exists.
  const v = view === "3d" && n3D > 0 ? "3d" : n2D > 0 ? "2d" : "3d";
  const is3d = v === "3d";

  const shown = all.filter((s) => (is3d ? s.alphaPrev : s.full));
  const n = shown.length;

  return (
    <div className="panel full inkseg">
      <h2>
        Ink predictions ({n} segment{n === 1 ? "" : "s"})
      </h2>
      <p className="txt" style={{ marginBottom: showToggle ? "10px" : "12px" }}>
        {is3d
          ? "Ink detected and painted back onto the rendered 3D surface of each segment."
          : "Ink detected on each flattened segment surface."}{" "}
        Thumbnails link to the full-resolution image; where available, each
        segment also links to its surface layers in Neuroglancer and its mesh.
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
          const fullUrl = is3d ? s.alpha : s.full;
          const { serviceUrl } = thumbnailUrls(previewUrl, 400, 400);
          const name = s.label || s.id;
          const label = `${name}${s.um ? ` · ${s.um}µm` : ""}`;
          const ng = s.layers ? neuroglancerUrl(s.layers, name) : null;
          return (
            <div key={s.id || i} className="inkcard">
              <a
                href={fullUrl}
                target="_blank"
                rel="noopener noreferrer"
                title={
                  is3d
                    ? "full-resolution 3D-ink render"
                    : "full-resolution ink prediction"
                }
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
                <a href={fullUrl} target="_blank" rel="noopener noreferrer">
                  {is3d ? "render ↗" : "ink ↗"}
                </a>
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
    </div>
  );
}
