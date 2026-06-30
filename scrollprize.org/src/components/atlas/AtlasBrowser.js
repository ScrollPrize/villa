import React, { useContext, useEffect, useState } from "react";
import BrowserOnly from "@docusaurus/BrowserOnly";
import Dashboard from "./Dashboard";
import ScrollGrid from "./ScrollGrid";
import AtlasRendererProvider, {
  AtlasRendererContext,
} from "./AtlasRendererProvider";
import useDarkModeGuard from "./useDarkModeGuard";

// Native React rebuild of the /data_browser index. Replaces the old component
// of the same import name (the mdx imports this default export). Mirrors the
// reference renderer's DOM/class structure (index.html) so the global ".atlas"
// CSS block styles it unchanged. The title/subtitle render server-side (for SEO
// and an immediate paint); the data-driven 3D grid is client-only.

const DATA_URL = "/data_browser/index.json";

// Pause/resume control for the shared 3D renderer. Must live inside the
// AtlasRendererProvider subtree because it reads paused/setPaused from context.
function PauseButton() {
  const ctx = useContext(AtlasRendererContext) || {};
  const { paused, setPaused } = ctx;
  return (
    <button
      type="button"
      className="btn"
      aria-pressed={paused ? "true" : "false"}
      onClick={() => setPaused && setPaused(!paused)}
    >
      {paused ? "▶ Resume spin" : "⏸ Pause spin"}
    </button>
  );
}

function AtlasBrowserInner() {
  // Keep the atlas dark-themed regardless of the site light/dark toggle.
  useDarkModeGuard();

  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(false);

  // Grid controls: free-text search, sort key, and filter toggles.
  const [query, setQuery] = useState("");
  const [sortKey, setSortKey] = useState("progress");
  const [typeFilter, setTypeFilter] = useState("all"); // all | scroll | fragment
  const [feat, setFeat] = useState({
    pred: false,
    ink: false,
    ink3d: false,
    ct: false,
  });
  const toggleFeat = (k) => setFeat((f) => ({ ...f, [k]: !f[k] }));

  useEffect(() => {
    let cancelled = false;
    const load = async () => {
      try {
        const res = await fetch(DATA_URL);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const json = await res.json();
        if (!cancelled) setData(json);
      } catch (err) {
        console.error("Failed to load data browser index:", err);
        if (!cancelled) setError(true);
      } finally {
        if (!cancelled) setLoading(false);
      }
    };
    load();
    return () => {
      cancelled = true;
    };
  }, []);

  if (loading) {
    return <p className="sub">Loading 3D models…</p>;
  }

  if (error || !data) {
    return (
      <p className="sub">
        Sorry, the data browser could not be loaded. Please try again later.
      </p>
    );
  }

  const scrolls = data.scrolls || [];
  const dashboard = data.dashboard || {};

  // ----- Filter + sort. Card keys stay `id` (see ScrollGrid) so the shared
  // WebGL canvas only churns for genuinely added/removed cards, not reorders. --
  const STAGE_ORDER = ["scanned", "segmented", "unrolled", "ink", "text"];
  const stageRank = (s) => {
    const st = s.stages || {};
    let r = 0;
    STAGE_ORDER.forEach((k, i) => {
      if (st[k]) r = i;
    });
    return r;
  };
  const segOf = (s) =>
    (s.progress && s.progress.segments != null
      ? s.progress.segments
      : s.n_segments) || 0;
  const score = (s) => (s.progress && s.progress.score) || 0;

  const q = query.trim().toLowerCase();
  const matches = (s) => {
    if (typeFilter === "scroll" && s.type !== "scroll") return false;
    if (typeFilter === "fragment" && s.type === "scroll") return false;
    if (feat.pred && !(s.n_predictions > 0)) return false;
    if (feat.ink && !(s.stages && s.stages.ink)) return false; // ink pipeline stage (matches the card badge + dashboard funnel)
    if (feat.ink3d && !s.hasInk3d) return false;
    if (feat.ct && !(s.mesh || s.volumeZarr)) return false;
    if (q) {
      const hay = `${s.id} ${s.display || ""} ${
        (s.content && s.content.work) || ""
      } ${s.desc || ""}`.toLowerCase();
      if (!hay.includes(q)) return false;
    }
    return true;
  };
  const sorters = {
    progress: (a, b) => stageRank(b) - stageRank(a) || score(b) - score(a),
    resolution: (a, b) => (a.min_px ?? 1e9) - (b.min_px ?? 1e9),
    segments: (a, b) => segOf(b) - segOf(a),
    predictions: (a, b) => (b.n_predictions || 0) - (a.n_predictions || 0),
    name: (a, b) =>
      String(a.id).localeCompare(String(b.id), undefined, { numeric: true }),
  };
  const view = scrolls.filter(matches).sort(sorters[sortKey] || sorters.progress);

  const featChip = (key, label) => (
    <button
      type="button"
      className={`chip${feat[key] ? " on" : ""}`}
      aria-pressed={feat[key] ? "true" : "false"}
      onClick={() => toggleFeat(key)}
    >
      {label}
    </button>
  );
  const typeChip = (val, label) => (
    <button
      type="button"
      className={`chip${typeFilter === val ? " on" : ""}`}
      aria-pressed={typeFilter === val ? "true" : "false"}
      onClick={() => setTypeFilter((t) => (t === val ? "all" : val))}
    >
      {label}
    </button>
  );

  return (
    <>
      <Dashboard dashboard={dashboard} />

      <AtlasRendererProvider>
        <div className="gridhead">
          <h2>Scrolls &amp; fragments</h2>
          <div className="gridctl">
            <input
              type="search"
              className="gridsearch"
              placeholder="Search name, work, description…"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              aria-label="Search samples"
            />
            <select
              className="gridsort"
              value={sortKey}
              onChange={(e) => setSortKey(e.target.value)}
              aria-label="Sort samples"
            >
              <option value="progress">Sort: Progress</option>
              <option value="resolution">Sort: Resolution</option>
              <option value="segments">Sort: Segments</option>
              <option value="predictions">Sort: Predictions</option>
              <option value="name">Sort: Name</option>
            </select>
            <PauseButton />
          </div>
        </div>

        <div className="gridfilters">
          {typeChip("scroll", "Scrolls")}
          {typeChip("fragment", "Fragments")}
          <span className="chipsep" aria-hidden="true" />
          {featChip("pred", "Predictions")}
          {featChip("ink", "Ink")}
          {featChip("ink3d", "3D ink")}
          {featChip("ct", "CT / 3D")}
          <span className="hint">
            {view.length} of {scrolls.length} samples
          </span>
        </div>

        {view.length ? (
          <ScrollGrid scrolls={view} />
        ) : (
          <p className="sub">No samples match these filters.</p>
        )}
      </AtlasRendererProvider>
    </>
  );
}

// Renders inside the standard docs page (the mdx supplies the page title and
// intro), so this only emits the dashboard + 3D grid that fill the content column.
export default function AtlasBrowser() {
  return (
    <div className="atlas">
      <BrowserOnly fallback={<p className="sub">Loading 3D models…</p>}>
        {() => <AtlasBrowserInner />}
      </BrowserOnly>
    </div>
  );
}
