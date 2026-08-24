import React from "react";
import { neuroglancerUrl, browseUrl } from "./dataAccess";

// ScansPanel — the per-scroll "Scans" card (below Data & access): one row per
// scan with its resolution, energy, source/beamline, and a Neuroglancer link
// for the OME-Zarr volume reconstructed from it. Promoted out of the Stats
// panel's compact link list; reuses the predictions table classes.

function fmt(v, suffix = "") {
  return v === null || v === undefined || v === "" ? "—" : `${v}${suffix}`;
}

export default function ScansPanel({ scans }) {
  if (!scans || !scans.length) return null;

  return (
    <div className="panel full scanspanel">
      <h2>Scans ({scans.length})</h2>
      <div className="tablewrap">
        <table className="predtable">
          <thead>
            <tr>
              <th>Scan</th>
              <th className="num">Resolution</th>
              <th className="num">Energy</th>
              <th>Source / beamline</th>
              <th>Links</th>
            </tr>
          </thead>
          <tbody>
            {scans.map((sc, i) => {
              const label =
                sc.name ||
                `${sc.px ? `${sc.px}µm ` : ""}${sc.energy ? `${sc.energy}keV ` : ""}${
                  sc.loc || ""
                }`.trim() ||
                "—";
              const ng = sc.volume ? neuroglancerUrl(sc.volume, label) : null;
              const files = sc.volume ? browseUrl(sc.volume) : null;
              return (
                <tr key={sc.name || i}>
                  <td>
                    <code>{label}</code>
                  </td>
                  <td className="num">{fmt(sc.px, " µm")}</td>
                  <td className="num">{fmt(sc.energy, " keV")}</td>
                  <td>{fmt(sc.loc)}</td>
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
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
