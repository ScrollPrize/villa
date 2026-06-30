import React from "react";
import { neuroglancerOverlayUrl, browseUrl } from "./dataAccess";

// PredictionsPanel — the volume-level "Predictions" table the old prebuilt
// atlas exposed (model · purpose · base volume · resolution · level · threshold
// · links). Data comes from `scroll.predictions` (built in genAtlasData.js from
// the public metadata.json): one row per surface-prediction / 3D-ink-detection
// zarr, each carrying the base volume's pixel size + energy.
//
// Links are built here (not in the data script) via ./dataAccess, exactly like
// the per-scroll scan links: "Overlay" opens the prediction volume in
// Neuroglancer; "Files" is the raw public HTTPS path. NOTE: the prediction
// volumes are Blosc-Zstd, which the zarr proxy can't yet decode, so the
// Neuroglancer links render only once the proxy gains Zstd support — the Files
// links work today. We surface that honestly in a footnote rather than hiding
// or faking the links.

const PURPOSE = {
  "ink-detection-3d": { label: "3D ink", cls: "ink3d" },
  "surface-prediction": { label: "surface", cls: "surface" },
};

function fmt(v) {
  return v === null || v === undefined || v === "" ? "—" : v;
}

export default function PredictionsPanel({ predictions }) {
  if (!predictions || !predictions.length) return null;
  const n = predictions.length;

  return (
    <div className="panel full predictions">
      <h2>
        Model predictions ({n})
      </h2>
      <p className="txt" style={{ marginBottom: "12px" }}>
        Volume-level machine-learning predictions over this scroll&apos;s CT
        data — surface geometry and, where available, 3D ink detection. Each row
        opens the prediction volume in Neuroglancer or links its raw files.
      </p>
      <div className="tablewrap">
        <table className="predtable">
          <thead>
            <tr>
              <th>Purpose</th>
              <th>Base volume</th>
              <th>Resolution</th>
              <th>Model</th>
              <th className="num">Level</th>
              <th className="num">Threshold</th>
              <th>Links</th>
            </tr>
          </thead>
          <tbody>
            {predictions.map((p, i) => {
              const meta = PURPOSE[p.purpose] || { label: p.purpose, cls: "" };
              const res =
                p.px != null
                  ? `${p.px} µm${p.energy != null ? ` · ${p.energy} keV` : ""}`
                  : "—";
              const ng = p.zarr
                ? neuroglancerOverlayUrl(p.baseZarr, p.zarr, {
                    base: p.baseVolume ? `${p.baseVolume} CT` : "CT volume",
                    pred: `${meta.label} prediction`,
                  })
                : null;
              const files = p.zarr ? browseUrl(p.zarr) : null;
              return (
                <tr key={`${p.purpose}-${p.baseVolume}-${i}`}>
                  <td>
                    <span className={`ppurpose ${meta.cls}`}>{meta.label}</span>
                  </td>
                  <td>
                    <code>{p.baseVolume || "—"}</code>
                  </td>
                  <td className="num">{res}</td>
                  <td>{p.model ? <code>{p.model}</code> : "—"}</td>
                  <td className="num">{fmt(p.level)}</td>
                  <td className="num">{fmt(p.threshold)}</td>
                  <td className="predlinks">
                    {ng ? (
                      <a href={ng} target="_blank" rel="noopener noreferrer">
                        Overlay ↗
                      </a>
                    ) : null}
                    {files ? (
                      <a href={files} target="_blank" rel="noopener noreferrer">
                        Files ↗
                      </a>
                    ) : null}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
