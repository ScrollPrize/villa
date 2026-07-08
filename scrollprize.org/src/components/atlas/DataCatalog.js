import React, { useState } from "react";
import { toHttp, neuroglancerUrl } from "./dataAccess";

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

  // Full-dataset ("samples") bucket paths for this item, and an optional
  // Neuroglancer deep-link for its OME-Zarr volume (see ./dataAccess.js).
  const samplesS3 = `s3://vesuvius-challenge/${scroll.id}/`;
  const samplesHttp = toHttp(samplesS3);
  const ctUrl = scroll.volumeZarr
    ? neuroglancerUrl(scroll.volumeZarr, `${scroll.display} CT`)
    : null;
  const licenses = scroll.licenses || [];
  // Optional per-license scope annotations curated in atlasOverlay.json
  // (license name -> which of this scroll's data it covers).
  const licenseScope = scroll.licenseScope || null;

  // Volume-level predictions are listed in their own Predictions table; here we
  // only summarize the count in the metadata list.
  const preds = scroll.predictions || [];
  const nSurface = preds.filter((p) => p.purpose === "surface-prediction").length;
  const nInk3d = preds.filter((p) => p.purpose === "ink-detection-3d").length;
  const predsTxt = [
    nSurface ? `${nSurface} surface` : null,
    nInk3d ? `${nInk3d} 3D-ink` : null,
  ]
    .filter(Boolean)
    .join(" · ");

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
  if (ctUrl) {
    links.push(
      <a
        key="ngct"
        className="dbtn"
        href={ctUrl}
        target="_blank"
        rel="noopener noreferrer"
      >
        CT in Neuroglancer ↗
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
        <dd>{energies.length ? `${energies.join(", ")} keV` : "—"}</dd>
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
        {predsTxt ? (
          <React.Fragment>
            <dt>Predictions</dt>
            <dd>{predsTxt}</dd>
          </React.Fragment>
        ) : null}
        <dt>License</dt>
        <dd>
          {licenses.length ? (
            licenses.map((l, i) => (
              <React.Fragment key={l.url}>
                {i ? " · " : ""}
                <a href={l.url} target="_blank" rel="noopener noreferrer">
                  {l.name}
                </a>
                {licenseScope && licenseScope[l.name] ? (
                  <span style={{ color: "var(--dim)" }}>
                    {" "}
                    ({licenseScope[l.name]})
                  </span>
                ) : null}
              </React.Fragment>
            ))
          ) : (
            <a href={LICENSE_URL} target="_blank" rel="noopener noreferrer">
              CC BY-NC 4.0
            </a>
          )}
        </dd>
      </dl>
      <PathRow label="HTTP" value={httpBase} />
      <PathRow label="S3" value={s3uri} />
      <PathRow label="Full dataset (HTTP)" value={samplesHttp} />
      <PathRow label="Full dataset (S3)" value={samplesS3} />
      <PathRow label=".volpkg" value={scroll.legacy} />
    </div>
  );
}
