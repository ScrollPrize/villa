import React from "react";
import { neuroglancerUrl, browseUrl } from "./dataAccess";

// ScansPanel — the per-scroll "Scans & volumes" card (below Data & access): one row per
// reconstructed VOLUME (the id people actually reference), with the scan-level
// facts (scan id, resolution, energy, source) stated once on the scan's first
// volume; further reconstructions of the same scan follow as separator-less
// rows with those cells blank, so the group reads as one band.
// Single-volume scans — the common case — read as a plain flat row.
// The scan id is shown short (the 14-digit id, dimmed): the long_id's µm/keV
// suffix would only duplicate the Resolution/Energy columns. Reuses the
// predictions table classes.

function fmt(v, suffix = "") {
  return v === null || v === undefined || v === "" ? "—" : `${v}${suffix}`;
}

export default function ScansPanel({ scans }) {
  if (!scans || !scans.length) return null;
  // Table row count: one row per volume, and a scan with no volume still
  // renders one (placeholder) row — mirrors the fallback in the body below.
  const nRows = scans.reduce((n, sc) => n + Math.max((sc.volumes || []).length, 1), 0);

  return (
    <div className="panel full scanspanel">
      <h2>Scans &amp; volumes ({nRows})</h2>
      <div className="tablewrap">
        <table className="predtable">
          <thead>
            <tr>
              <th>Scan</th>
              <th className="num">Resolution</th>
              <th className="num">Energy</th>
              <th>Source / beamline</th>
              <th>Volume</th>
              <th>Links</th>
            </tr>
          </thead>
          <tbody>
            {scans.map((sc, i) => {
              const ngLabel = sc.name || sc.id || "scan";
              // Older snapshots carry a single `volume` zarr instead of the
              // `volumes` list — synthesize a one-entry list so both render.
              const vols =
                sc.volumes && sc.volumes.length
                  ? sc.volumes
                  : [{ id: null, zarr: sc.volume || null }];
              return vols.map((v, j) => {
                const ng = v.zarr ? neuroglancerUrl(v.zarr, ngLabel) : null;
                const files = v.zarr ? browseUrl(v.zarr) : null;
                // Later volumes of a multi-volume scan leave the scan-level
                // cells blank and drop the separator line above, so the group
                // reads as one band with the scan facts stated once at its top.
                const cont = j > 0;
                return (
                  <tr className={cont ? "scancont" : undefined} key={v.id || `${sc.id || sc.name || i}-${j}`}>
                    <td>
                      {cont ? null : <code>{sc.id || sc.name || "—"}</code>}
                    </td>
                    <td className="num">{cont ? null : fmt(sc.px, " µm")}</td>
                    <td className="num">{cont ? null : fmt(sc.energy, " keV")}</td>
                    <td>{cont ? null : fmt(sc.loc)}</td>
                    <td>
                      <code>{v.id || "—"}</code>
                    </td>
                    <td className="predlinks">
                      {ng ? (
                        <a href={ng} target="_blank" rel="noopener noreferrer">
                          Neuroglancer ↗
                        </a>
                      ) : null}
                      {files ? (
                        <a href={files} target="_blank" rel="noopener noreferrer">
                          Files ↗
                        </a>
                      ) : null}
                      {!ng && !files ? "—" : null}
                    </td>
                  </tr>
                );
              });
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
