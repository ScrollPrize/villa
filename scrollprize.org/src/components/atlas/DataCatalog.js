import React, { useState } from "react";

// DataCatalog — the "Data & access" panel for a scroll detail page.
// Ported from the reference renderer (ref/scroll.html ~104-127): a set of
// quick-link buttons, a technical-metadata definition list, and copy-pasteable
// HTTP / S3 / .volpkg paths. Class names match the reference DOM exactly so the
// global `.atlas` CSS block styles it.

// Standard Vesuvius Challenge open-data license wording.
const LICENSE_URL = "https://dl.ash2txt.org/LICENSE.txt";

// A path row with a copy-to-clipboard button. Renders nothing if no value.
function PathRow({ label, value }) {
  const [copied, setCopied] = useState(false);
  if (!value) return null;

  const onCopy = () => {
    if (typeof navigator !== "undefined" && navigator.clipboard) {
      navigator.clipboard.writeText(value).then(() => {
        setCopied(true);
        setTimeout(() => setCopied(false), 1200);
      });
    }
  };

  return (
    <div className="pathrow">
      <span className="plab">{label}</span>
      <code>{value}</code>
      <button className="copybtn" type="button" onClick={onCopy}>
        {copied ? "copied ✓" : "copy"}
      </button>
    </div>
  );
}

export default function DataCatalog({ scroll }) {
  if (!scroll) return null;

  const progress = scroll.progress || {};

  // Derive copy-pasteable paths from the bucket URL (ref lines 105-106).
  const httpBase = scroll.bucketUrl
    ? scroll.bucketUrl.replace("/index.html#", "/")
    : "";
  const s3uri = httpBase.replace(
    "https://vesuvius-challenge-open-data.s3.amazonaws.com/",
    "s3://vesuvius-challenge-open-data/",
  );

  // Quick-link buttons (ref lines 108-112).
  const links = [];
  if (progress.wkUrl) {
    links.push(
      <a
        key="wk"
        className="dbtn"
        href={progress.wkUrl}
        target="_blank"
        rel="noopener noreferrer"
      >
        webknossos viewer ↗
      </a>,
    );
  }
  if (scroll.bucketUrl) {
    links.push(
      <a
        key="browse"
        className="dbtn"
        href={scroll.bucketUrl}
        target="_blank"
        rel="noopener noreferrer"
      >
        Browse files ↗
      </a>,
    );
  }
  if (scroll.legacy) {
    links.push(
      <a
        key="volpkg"
        className="dbtn"
        href={scroll.legacy}
        target="_blank"
        rel="noopener noreferrer"
      >
        .volpkg ↗
      </a>,
    );
  }

  // Metadata rows (ref lines 115-123).
  const energies = scroll.energies || [];
  const locations = scroll.locations || [];
  const segments = progress.segments != null ? progress.segments : scroll.n_segments;
  const segmentsTxt =
    segments != null ? Number(segments).toLocaleString() : "—";
  const patchesTxt = progress.patches
    ? ` / ${Number(progress.patches).toLocaleString()} patches`
    : "";

  return (
    <div className="panel full catalog">
      <h2>Data &amp; access</h2>
      <div className="dbtns">
        {links.length > 0 ? (
          links
        ) : (
          <span style={{ color: "var(--dim)", fontSize: "12.5px" }}>
            no scroll-specific links yet
          </span>
        )}
      </div>
      <dl className="meta">
        <dt>Voxel size (min)</dt>
        <dd>{scroll.min_px ? `${scroll.min_px} µm` : "—"}</dd>
        <dt>Energies</dt>
        <dd>{(energies.join(", ") || "—") + " keV"}</dd>
        <dt>Source / beamline</dt>
        <dd>{locations.join(", ") || "—"}</dd>
        <dt>Scans / volumes</dt>
        <dd>
          {scroll.n_scans} / {scroll.n_volumes}
        </dd>
        <dt>Segments</dt>
        <dd>
          {segmentsTxt}
          {patchesTxt}
        </dd>
        <dt>Formats</dt>
        <dd>CT volumes (TIFF stacks · OME-Zarr) · surface segments</dd>
        <dt>License</dt>
        <dd>
          CC BY-NC 4.0 ·{" "}
          <a href={LICENSE_URL} target="_blank" rel="noopener noreferrer">
            terms
          </a>
        </dd>
      </dl>
      <PathRow label="HTTP" value={httpBase} />
      <PathRow label="S3" value={s3uri} />
      <PathRow label=".volpkg" value={scroll.legacy} />
    </div>
  );
}
