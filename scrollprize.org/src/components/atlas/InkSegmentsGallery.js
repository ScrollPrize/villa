import React from "react";
import { thumbnailUrls } from "./dataAccess";

// InkSegmentsGallery — the per-segment ink-detection listing the old atlas had:
// every segment that has an ink-detection prediction, as a lazy-loaded grid of
// small Thumbor thumbnails, each linking to the full-resolution image.
//
// Data comes from `scroll.inkSegments` (built in scripts/genAtlasData.js from
// the public metadata.json). Each entry is { id, um, full, preview }. We
// thumbnail the (large) downsampled `preview` through the Thumbor service so the
// grid loads promptly. Captions are segment id + resolution only — no internal
// model codenames.

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
        full-resolution image.
      </p>
      <div className="inkgrid">
        {segments.map((s, i) => {
          const { serviceUrl } = thumbnailUrls(s.preview, 400, 400);
          const name = s.label || s.id;
          const label = `${name}${s.um ? ` · ${s.um}µm` : ""}`;
          return (
            <a
              key={s.id || i}
              className="inkcard"
              href={s.full}
              target="_blank"
              rel="noopener noreferrer"
              title={label}
            >
              <img
                src={serviceUrl || s.preview}
                alt={`${display || ""} ink-detection prediction — segment ${name}`}
                loading="lazy"
                onError={(e) => {
                  e.currentTarget.style.display = "none";
                }}
              />
              <span className="inklabel">{label}</span>
            </a>
          );
        })}
      </div>
    </div>
  );
}
