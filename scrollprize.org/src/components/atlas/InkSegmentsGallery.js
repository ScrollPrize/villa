import React from "react";
import { thumbnailUrls, neuroglancerUrl, toHttp } from "./dataAccess";

// InkSegmentsGallery — the per-segment listing the old atlas had: every segment
// with an ink-detection prediction, as a lazy-loaded grid of small Thumbor
// thumbnails. Each card links to the full-resolution ink image, and (where the
// metadata has them) to the segment's flattened layers in Neuroglancer and its
// mesh download.
//
// Data comes from `scroll.inkSegments` (built in scripts/genAtlasData.js from
// the public metadata.json): { id, label, um, full, preview, layers, mesh }.
// Captions are segment id + resolution only — no internal model codenames.

export default function InkSegmentsGallery({ segments, display }) {
  if (!segments || !segments.length) return null;
  const n = segments.length;

  return (
    <div className="panel full inkseg">
      <h2>
        Ink-detection predictions ({n} segment{n === 1 ? "" : "s"})
      </h2>
      <p className="txt" style={{ marginBottom: "12px" }}>
        Every segment with an ink-detection prediction. Thumbnails link to the
        full-resolution image; where available, each segment also links to its
        surface layers in Neuroglancer and its mesh.
      </p>
      <div className="inkgrid">
        {segments.map((s, i) => {
          const { serviceUrl } = thumbnailUrls(s.preview, 400, 400);
          const name = s.label || s.id;
          const label = `${name}${s.um ? ` · ${s.um}µm` : ""}`;
          const ng = s.layers ? neuroglancerUrl(s.layers, name) : null;
          return (
            <div key={s.id || i} className="inkcard">
              <a
                href={s.full}
                target="_blank"
                rel="noopener noreferrer"
                title="full-resolution ink prediction"
              >
                <img
                  src={serviceUrl || s.preview}
                  alt={`${display || ""} ink-detection prediction — segment ${name}`}
                  loading="lazy"
                  onError={(e) => {
                    e.currentTarget.style.display = "none";
                  }}
                />
              </a>
              <span className="inklabel">{label}</span>
              <span className="inklinks">
                <a href={s.full} target="_blank" rel="noopener noreferrer">
                  ink ↗
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
