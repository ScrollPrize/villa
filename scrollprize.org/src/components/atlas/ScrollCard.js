import React from "react";
import Link from "@docusaurus/Link";
import PipelineStepper from "./PipelineStepper";
import useAtlasCard from "./useAtlasCard";

// A single scroll/fragment card. Ported from the reference renderer card markup
// (index.html ~233-247). The whole card is a Docusaurus <Link> to the sample's
// detail page. JSX auto-escapes text, so the reference esc() calls become plain
// interpolation; only the markdown-stripping logic from mdClean()/caption() is
// reproduced here.

// Boilerplate sentences we drop from the auto-generated caption so cards show a
// distinctive line rather than the same "rolled Herculaneum papyrus…" preamble.
const BOIL =
  /rolled Herculaneum papyrus|carbonized during the eruption|scanned at the|Mount Vesuvius|^see also/i;

// Distinctive one-liner for a sample: a known work title, else the
// non-boilerplate tail of its description, else nothing. Ported from the
// reference caption() (index.html ~204-214) minus the esc() (JSX escapes).
function caption(scroll) {
  const w = scroll.content && scroll.content.work;
  if (w && !/not yet identified/i.test(w)) return "📖 " + w;

  let d = String(scroll.desc || "")
    .replace(/\[([^\]]+)\]\([^)]*\)/g, "$1")
    .replace(/[*_`#]+/g, "")
    .replace(/\s+/g, " ")
    .trim();
  if (!d) return "";

  // Protect "PHerc." dots from the sentence splitter, then restore.
  d = d.replace(/PHerc\./g, "PHerc§");
  const keep = d
    .split(/(?<=\.)\s+/)
    .map((x) => x.replace(/PHerc§/g, "PHerc."))
    .filter((x) => x && !BOIL.test(x.replace(/PHerc\./g, "PHerc")));
  const out = keep
    .join(" ")
    .replace(/^PHerc\.\s*/, "")
    .trim();
  return out.length > 15 ? out : "";
}


// Distinct scan pixel sizes, finest first — "2.4 µm / 7.91 µm / …".
function pixelSizesText(scroll) {
  const px = [...new Set((scroll.scans || []).map((s) => s.px).filter((v) => v != null))].sort(
    (a, b) => a - b,
  );
  if (!px.length) return scroll.min_px ? `${scroll.min_px} µm` : "—";
  return px.map((v) => `${v} µm`).join(" / ");
}

export default function ScrollCard({ scroll }) {
  // Registers this card's 3D viewport with the shared renderer. No-op when the
  // sample has no mesh; we only attach the ref to the .view in that case.
  const viewRef = useAtlasCard(scroll.id, scroll.mesh);

  const readings = scroll.readings;
  const progress = scroll.progress || {};
  const mesh = scroll.mesh;
  const photo = scroll.photo;

  // Prediction signals (derived in genAtlasData.js from metadata.json).
  const hasInk3d = !!scroll.hasInk3d;
  // "ink" reflects the pipeline stage (matches the dashboard's "with
  // ink-detection results" count), not just scrolls whose per-segment ink
  // images happen to be published in the open-data metadata.
  const hasInk = !!(scroll.stages && scroll.stages.ink);
  const nInk = scroll.n_inkSegments || 0;
  const nAlpha = scroll.n_alpha || 0;

  // Honest image count: 3D model + photo + 2D-ink + 3D-ink renders.
  const imgN = (mesh ? 1 : 0) + (photo ? 1 : 0) + nInk + nAlpha;

  // Segment count: the sample's factual segment count from the S3 metadata.
  const segN = scroll.n_segments || 0;

  const isScroll = scroll.type === "scroll";
  const label = scroll.label || scroll.id;
  const nick = scroll.display && scroll.display !== scroll.id ? scroll.display : null;
  const cap = caption(scroll);

  // The 3D / image viewport.
  let view;
  if (mesh) {
    view = (
      <div className="view" ref={viewRef}>
        <div className="ph">loading 3D…</div>
      </div>
    );
  } else if (photo) {
    view = (
      <div className="view">
        <img
          className="fragimg"
          src={photo}
          alt={`Photo of ${label}`}
          loading="lazy"
        />
      </div>
    );
  } else {
    view = (
      <div className="view">
        <div className="ph">no 3D model</div>
      </div>
    );
  }

  return (
    <Link
      className="card"
      to={`/data_browser/${scroll.id}`}
      aria-label={`${label}${nick ? ` (${nick})` : ""} — details`}
    >
      <div className="chead">
        <span className="name">
          {label}
          {hasInk3d ? (
            <span
              className="pb ink3d"
              title="3D ink prediction — detected ink mapped onto the rendered papyrus surface"
            >
              3D ink
            </span>
          ) : null}
          {hasInk ? (
            <span
              className="rd"
              title="Ink-detection results available for this scroll"
            >
              ink
            </span>
          ) : null}
        </span>
        <span className="pherc">
          {nick ? nick : ""}
          {nick && imgN ? " · " : ""}
          {imgN ? `🖼 ${imgN}` : ""}
        </span>
      </div>

      {view}

      <PipelineStepper stages={scroll.stages} />

      <div className="stats">
        <span className={`type ${isScroll ? "" : "fragment"}`}>
          {scroll.type || "sample"}
        </span>
        <dl>
          <dt>Status</dt>
          <dd>
            <b style={{ color: "var(--accent)" }}>
              {progress.textFound || "Scanned"}
            </b>
          </dd>
          <dt>Segments</dt>
          <dd>{segN.toLocaleString()}</dd>
          <dt>Available pixel sizes</dt>
          <dd>{pixelSizesText(scroll)}</dd>
        </dl>
        {cap ? <p className="desc">{cap}</p> : null}
      </div>
    </Link>
  );
}
