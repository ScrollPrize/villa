import React from "react";

// The progress-overview dashboard. Ported from the reference renderer
// (index.html ~162-189): a row of headline tiles and the 5-stage pipeline
// funnel with a note. (The featured-reading marquee and "what's new" timeline
// panels were removed.)

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
            <div className={`stage st${i + 1}`} key={f.key || i}>
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
          Counts are scrolls/fragments with results at each stage. Ink detection
          is also run on flat fragments, so it can exceed the “unrolled” count —
          geometric unrolling, not ink, is the current bottleneck.
        </div>
      </div>
    </section>
  );
}
