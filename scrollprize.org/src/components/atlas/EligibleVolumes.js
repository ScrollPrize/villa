import React from "react";
import Link from "@docusaurus/Link";
import Details from "@theme/Details";
import useAtlasData from "./useAtlasData";
import { neuroglancerUrl } from "./dataAccess";
import eligibility from "@site/src/data/prizeEligibility.json";

// EligibleVolumes — the prize-eligible volume list for one prize, rendered as
// a collapsed <details> block on the prizes page.
//
// src/data/prizeEligibility.json is the source of truth: a map from prize slug
// (the `id` in docs/34_prizes.md's `prizes:` frontmatter) to `{scroll, volume}`
// pairs, where both ids match the data browser's. `scroll` is redundant given
// the volume id but keeps the file human-readable and lets consumers detect a
// mismatched pair. Display names, scan names, and neuroglancer links are all
// resolved against the same index the data catalog uses (live
// metadata.min.json, falling back to the bundled build-time snapshot), so this
// list can never drift from the catalog. Until the index loads (or for a pair
// the index doesn't know), each entry falls back to its raw ids.

export default function EligibleVolumes({ prize, summary = "Eligible scroll volumes" }) {
  const { index } = useAtlasData();
  const entries = eligibility[prize] || [];
  const byId = new Map(((index && index.scrolls) || []).map((s) => [s.id, s]));

  // @theme/Details is the card-styled component MDX substitutes for a raw
  // <details> in markdown — using it keeps this block visually identical to
  // the page's other collapsibles.
  return (
    <Details
      summary={
        <summary>
          {summary} ({entries.length})
        </summary>
      }
    >
      <ol>
        {entries.map(({ scroll, volume }) => {
          const s = byId.get(scroll);
          let scanName = null;
          let ng = null;
          for (const scan of (s && s.scans) || []) {
            for (const v of scan.volumes || []) {
              if (v.id === volume) {
                scanName = scan.name || scan.id;
                ng = neuroglancerUrl(v.zarr, `${s.display} ${v.id}`);
              }
            }
          }
          return (
            <li key={`${scroll}/${volume}`}>
              <Link to={`/data_browser/${scroll}`}>{(s && s.label) || scroll}</Link>
              {" — "}
              {ng ? (
                <a href={ng} target="_blank" rel="noopener noreferrer">
                  {scanName}
                </a>
              ) : (
                <code>{volume}</code>
              )}
            </li>
          );
        })}
      </ol>
    </Details>
  );
}
