import React from "react";
import Link from "@docusaurus/Link";

// The progress-overview dashboard. Ported from the reference renderer
// (index.html ~162-189): a row of headline tiles, the 5-stage pipeline funnel
// with a note, and a row pairing the featured-reading marquee with the "what's
// new" timeline. This is the PUBLIC page, so the reference's "embargoed" pill
// and "visible only on this protected site" line are intentionally omitted.

// Rewrite a marquee/reading src (e.g. "readings/paris4_ink.jpg" or a bare
// "paris4_ink.jpg") to its public path under the Docusaurus static dir.
function readingUrl(src) {
  const base = String(src || "").split("/").pop();
  return `/img/data_browser/readings/${base}`;
}

export default function Dashboard({ dashboard, timeline }) {
  const dash = dashboard || {};
  const tiles = dash.tiles || [];
  const funnel = dash.funnel || [];
  const tl = timeline || [];

  // Bar widths are relative to the largest funnel count.
  const fmax = Math.max(...funnel.map((f) => f.count || 0), 1);

  const mq = dash.marquee;

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

      <div className="row2">
        {mq ? (
          <div className="panel marq">
            <h2>Featured reading</h2>
            <Link to={`/data_browser/${mq.id}`}>
              <img
                src={readingUrl(mq.src)}
                alt={`Ink-detection reading from ${mq.display}`}
                loading="lazy"
              />
            </Link>
            <div className="cap">
              {mq.id}
              {mq.display && mq.display !== mq.id ? ` (${mq.display})` : ""} —{" "}
              {mq.cap || ""} ·{" "}
              <Link className="lk" to={`/data_browser/${mq.id}`}>
                open {mq.id} →
              </Link>
            </div>
          </div>
        ) : null}

        <div className="panel">
          <h2>What's new</h2>
          <ul className="tl">
            {tl.map((e, i) => (
              <li className={e.future ? "future" : ""} key={i}>
                <span className="dt">
                  {e.future ? "soon · " : ""}
                  {e.date}
                </span>
                <span>
                  <span className="tt">{e.title}</span>
                  <br />
                  <span className="bl">{e.blurb || ""}</span>
                </span>
              </li>
            ))}
          </ul>
        </div>
      </div>
    </section>
  );
}
