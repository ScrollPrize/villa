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
  const timeline = data.timeline || [];
  const meshCount = scrolls.filter((s) => s.mesh).length;

  return (
    <>
      <Dashboard dashboard={dashboard} timeline={timeline} />

      <AtlasRendererProvider>
        <div className="gridhead">
          <h2>Scrolls &amp; fragments</h2>
          <div style={{ display: "flex", gap: "12px", alignItems: "center" }}>
            <span className="hint">
              {scrolls.length} samples · {meshCount} with 3D · sorted by progress
            </span>
            <PauseButton />
          </div>
        </div>
        <ScrollGrid scrolls={scrolls} />
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
