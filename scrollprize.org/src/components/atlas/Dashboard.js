import React from "react";

// The progress-overview dashboard. Ported from the reference renderer
// (index.html ~162-189): a row of headline tiles and the 4-stage pipeline
// funnel with a note. (The featured-reading marquee and "what's new" timeline
// panels were removed.)

// Stage -> colour-class map, so each funnel stage keeps a stable colour by
// meaning rather than position. Segmentation + unrolling are one "Segmented"
// stage; per-scroll unrolling is shown as "% unrolled".
const STAGE_CLS = { scanned: "st1", segmented: "st2", ink: "st4", text: "st5" };

export default function Dashboard({ dashboard }) {
  const dash = dashboard || {};
  const tiles = dash.tiles || [];
  const funnel = dash.funnel || [];

  // Bar widths are relative to the largest funnel count.
  const fmax = Math.max(...funnel.map((f) => f.count || 0), 1);

  return (
    <section className="dash" aria-label="Progress overview">
      <div className="tiles">
        {tiles.map((t, i) => (
          <div className="tile" key={i}>
            <div className="big">{t.big}</div>
            <div className="lab">{t.label}</div>
            <div className="sub">{t.sub || ""}</div>
          </div>
        ))}
      </div>

      <div className="panel">
        <h2>The pipeline — scanned → read</h2>
        <div className="funnel">
          {funnel.map((f, i) => (
            <div className={`stage ${STAGE_CLS[f.key] || `st${i + 1}`}`} key={f.key || i}>
              <div className="cnt">{f.count}</div>
              <div className="nm">{f.label}</div>
              <div className="ds">{f.desc || ""}</div>
              <div className="bar">
                <i
                  style={{
                    width: `${Math.round((100 * (f.count || 0)) / fmax)}%`,
                  }}
                />
              </div>
            </div>
          ))}
        </div>
        <div className="funnel-note">
          Counts are scrolls/fragments that reached each stage. The Segmented
          stage covers surface tracing and unrolling; how far each scroll is
          unrolled is shown per scroll as its “% unrolled.”
        </div>
      </div>
    </section>
  );
}
