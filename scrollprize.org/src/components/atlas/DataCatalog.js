import React, { useEffect, useRef, useState } from "react";
import { neuroglancerUrl } from "./dataAccess";
import { scanConfigurations, webknossosDatasets } from "./buildIndex";

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

// Picker for a quick link that covers several volumes — the raw-CT OME-Zarr
// volumes in Neuroglancer, the scroll's WEBKNOSSOS datasets: a dbtn that opens
// a menu of one link per volume, each labeled with its name + resolution (and
// energy, where the metadata states one). Click-away closes it.
function VolumeDropdown({ label, items }) {
  const [open, setOpen] = useState(false);
  const ref = useRef(null);
  useEffect(() => {
    if (!open) return undefined;
    const onDown = (e) => {
      if (ref.current && !ref.current.contains(e.target)) setOpen(false);
    };
    document.addEventListener("mousedown", onDown);
    return () => document.removeEventListener("mousedown", onDown);
  }, [open]);

  return (
    <span className="dbtn-dd" ref={ref}>
      <button
        type="button"
        className="dbtn"
        aria-haspopup="menu"
        aria-expanded={open}
        onClick={() => setOpen((o) => !o)}
      >
        {label} ({items.length}) ▾
      </button>
      {open ? (
        <div className="dbtn-menu" role="menu">
          {items.map((it) => (
            <a
              key={it.key}
              role="menuitem"
              href={it.href}
              target="_blank"
              rel="noopener noreferrer"
              onClick={() => setOpen(false)}
            >
              {it.title}
              {it.detail ? <span className="ddmeta"> {it.detail}</span> : null} ↗
            </a>
          ))}
        </div>
      ) : null}
    </span>
  );
}

function configText(c) {
  const parts = [
    c.px != null ? `${c.px} µm` : null,
    c.energy != null ? `${c.energy} keV` : null,
    c.loc || null,
    c.n > 1 ? `${c.n} scans` : null,
  ].filter(Boolean);
  return parts.length ? parts.join(" · ") : "unspecified";
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

  // Raw-CT OME-Zarr volumes → Neuroglancer, and the scroll's WEBKNOSSOS
  // datasets. One volume keeps the plain button; several get a picker
  // (VolumeDropdown).
  const ctVolumes = (scroll.ctVolumes || []).filter((v) => v.zarr);
  const ctItems = ctVolumes.map((v) => ({
    key: v.id,
    href: neuroglancerUrl(v.zarr, `${scroll.display} ${v.id}`),
    title: v.id,
    detail: [
      v.px != null ? `${v.px} µm` : null,
      v.energy != null ? `${v.energy} keV` : null,
    ]
      .filter(Boolean)
      .join(" · "),
  }));
  const wkItems = webknossosDatasets(progress).map((d, i) => ({
    key: d.url || i,
    href: d.url,
    title: d.name || "webknossos viewer",
    detail: d.px != null ? `${d.px} µm` : "",
  }));
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
  if (wkItems.length > 1) {
    links.push(<VolumeDropdown key="wk" label="webknossos" items={wkItems} />);
  } else if (wkItems.length === 1) {
    links.push(
      <a
        key="wk"
        className="dbtn"
        href={wkItems[0].href}
        target="_blank"
        rel="noopener noreferrer"
      >
        webknossos viewer ↗
      </a>,
    );
  }
  if (ctItems.length > 1) {
    links.push(
      <VolumeDropdown key="ngct" label="CT in Neuroglancer" items={ctItems} />,
    );
  } else if (ctItems.length === 1) {
    links.push(
      <a
        key="ngct"
        className="dbtn"
        href={ctItems[0].href}
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
  const configs = scanConfigurations(scroll.scans);
  const segments = scroll.n_segments;
  const segmentsTxt =
    segments != null ? Number(segments).toLocaleString() : "—";

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
        <dt>Scan configurations</dt>
        <dd>
          {configs.length
            ? configs.map((c) => (
                <div key={`${c.px}|${c.energy}|${c.loc}`}>{configText(c)}</div>
              ))
            : "—"}
        </dd>
        <dt>Scans / volumes</dt>
        <dd>
          {scroll.n_scans} / {scroll.n_volumes}
        </dd>
        <dt>Segments</dt>
        <dd>{segmentsTxt}</dd>
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
      <PathRow label=".volpkg" value={scroll.legacy} />
    </div>
  );
}
