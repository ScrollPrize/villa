import React from "react";

// The progress-overview dashboard: a row of headline tiles. (The 4-stage
// pipeline funnel was dropped — it repeated the tiles' numbers; the marquee
// and "what's new" timeline panels were removed earlier.)

export default function Dashboard({ dashboard }) {
  const dash = dashboard || {};
  const tiles = dash.tiles || [];

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
    </section>
  );
}
