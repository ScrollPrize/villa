import { useEffect, useState } from "react";
import { usePluginData } from "@docusaurus/useGlobalData";
import { buildIndex } from "./buildIndex";
import { rewrites } from "./dataAccess";
import overlay from "@site/src/data/atlasOverlay.json";
import meshManifest from "@site/static/img/data_browser/meshes/manifest.json";
// Byte sizes of the multi-variant downsampled renders, generated alongside
// index.json by scripts/genAtlasData.js. Passing the same map here keeps the
// live derivation's primary-render choice identical to the build-time one
// (a variant published after the last build simply falls back to the
// finest-um-first order until the next build refreshes this file).
import renderSizes from "@site/static/data_browser/renderSizes.json";

// useAtlasData — load the data-browser index at runtime.
//
// Prefers the LIVE, always-current factual projection published next to the
// public metadata (`metadata.min.json`, a small field-subset of metadata.json),
// merged in-browser with the checked-in curated overlay via the shared
// buildIndex() — the exact same derivation scripts/genAtlasData.js runs at build
// time, so live and build-time data are byte-identical for the same inputs.
//
// Falls back to the build-time snapshot (static/data_browser/index.json) when
// the live fetch fails — S3/CORS/network error, or a build that predates the
// projection being published. This is why shipping the website change before
// the producer starts publishing metadata.min.json is safe: it simply keeps
// serving the (build-time) bundled index until the live file appears.

const META_MIN_URL =
  "https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/metadata.min.json";
const BUNDLED_URL = "/data_browser/index.json";

// Resolve the merged index, preferring live metadata.min.json over the bundled
// build-time snapshot. Returns { index, source: "live" | "bundled" }.
//
// `routedIds` (optional): the ids that have a static detail route this build.
// When given, live samples are restricted to that set BEFORE buildIndex, so a
// brand-new scroll (present in metadata.min.json but not yet built into a page)
// is excluded from BOTH the grid and the derived dashboard counts — keeping them
// consistent and avoiding a card that links to a 404. It surfaces at the next
// build. Omit it (e.g. from a detail page, which only looks up its own routed
// id) to skip filtering.
export async function loadAtlasIndex(routedIds) {
  try {
    const res = await fetch(META_MIN_URL);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const meta = await res.json();
    let samples = (meta && meta.samples) || {};
    if (!Object.keys(samples).length) throw new Error("no samples in metadata.min.json");
    if (routedIds && routedIds.length) {
      const allow = new Set(routedIds);
      samples = Object.fromEntries(
        Object.entries(samples).filter(([id]) => allow.has(id))
      );
    }
    return {
      index: buildIndex(samples, { overlay, meshManifest, rewrites, renderSizes }),
      source: "live",
    };
  } catch (liveErr) {
    // Live projection unavailable — fall back to the build-time snapshot.
    const res = await fetch(BUNDLED_URL);
    if (!res.ok) throw new Error(`bundled index HTTP ${res.status}`);
    return { index: await res.json(), source: "bundled" };
  }
}

// React hook wrapping loadAtlasIndex() with loading/error state. Client-only
// (the fetch runs in an effect), so callers should already be browser-scoped.
// Reads the atlas-data plugin's routedIds so the live grid/dashboard only
// include scrolls that have a detail page this build.
export default function useAtlasData() {
  const routedIds = (usePluginData("atlas-data") || {}).routedIds;
  const [state, setState] = useState({
    index: null,
    source: null,
    loading: true,
    error: false,
  });
  useEffect(() => {
    let cancelled = false;
    loadAtlasIndex(routedIds)
      .then(({ index, source }) => {
        if (!cancelled) setState({ index, source, loading: false, error: false });
      })
      .catch((err) => {
        console.error("Failed to load data browser index:", err);
        if (!cancelled)
          setState({ index: null, source: null, loading: false, error: true });
      });
    return () => {
      cancelled = true;
    };
  }, [routedIds]);
  return state;
}
