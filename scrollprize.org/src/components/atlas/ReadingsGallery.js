import React from "react";

// ReadingsGallery — the "Ink detection & renders" panel.
// Ported from the reference renderer (ref/scroll.html ~142-152). Unlike the
// reference, embargoed readings are stripped to null upstream, so anything we
// receive here is PUBLIC: we deliberately render NO embargo banner. Each image
// `src` is already an absolute `/img/data_browser/readings/...` path.

export default function ReadingsGallery({ readings, display }) {
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
            <img src={im.src} alt={alt} loading="lazy" />
            <div className="cap">
              {cap ? `${cap} · ` : ""}
              <a href={im.src} target="_blank" rel="noopener noreferrer">
                full image ↗
              </a>
            </div>
          </React.Fragment>
        );
      })}
    </div>
  );
}
