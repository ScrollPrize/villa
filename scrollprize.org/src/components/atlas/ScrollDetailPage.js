import React from "react";
import Layout from "@theme/Layout";
import Head from "@docusaurus/Head";
import Link from "@docusaurus/Link";
import BrowserOnly from "@docusaurus/BrowserOnly";
import JsonLd from "@site/src/components/JsonLd";
import ScrollViewer from "./ScrollViewer";
import DataCatalog from "./DataCatalog";
import ReadingsGallery from "./ReadingsGallery";
import InkSegmentsGallery from "./InkSegmentsGallery";
import useDarkModeGuard from "./useDarkModeGuard";

// ScrollDetailPage — the per-scroll detail route for the rebuilt /data_browser.
// This is a Docusaurus ROUTE component: a plugin injects one merged `scroll`
// object as props at build time. The DOM here reproduces the reference detail
// page (ref/scroll.html ~154-182) and keeps its exact class names so the global
// `.atlas` CSS block styles it. Everything is server-rendered except the
// interactive 3D viewer (wrapped in <BrowserOnly>).

const SITE = "https://scrollprize.org";
const LICENSE_URL = "https://dl.ash2txt.org/LICENSE.txt";

// Pipeline stepper config (ref lines 130-131). The detail page uses the
// `.steps2`/`.plab2` classes (distinct from the index card's `.steps`).
const STAGE_KEYS = ["scanned", "segmented", "unrolled", "ink", "text"];
const STAGE_LABEL = {
  scanned: "Scanned",
  segmented: "Segmented",
  unrolled: "Unrolled",
  ink: "Ink detected",
  text: "Text recovered",
};

// Strip light markdown to plain text: drop images, unwrap links, remove
// emphasis/heading/code markers. Used wherever we render desc/note/summary.
function stripMarkdown(md) {
  if (!md) return "";
  return String(md)
    .replace(/!\[[^\]]*\]\([^)]*\)/g, "") // ![alt](url) -> ""
    .replace(/\[([^\]]*)\]\([^)]*\)/g, "$1") // [text](url) -> text
    .replace(/[*_`#>]/g, "") // emphasis / heading / code / quote markers
    .replace(/\s+/g, " ")
    .trim();
}

// Build a ~155-char plain-text meta description.
function metaDescription(scroll) {
  const content = scroll.content || {};
  const source = content.summary || scroll.desc || "";
  const plain = stripMarkdown(source);
  if (plain.length <= 155) return plain;
  return plain.slice(0, 152).trimEnd() + "…";
}

export default function ScrollDetailPage(props) {
  const scroll = props.scroll;
  useDarkModeGuard();

  if (!scroll) {
    return (
      <Layout title="Unknown scroll — Vesuvius Challenge">
        <div className="atlas atlas-detail">
          <div className="wrap">
            <Link className="back" to="/data_browser">
              ← all scrolls
            </Link>
            <p>Unknown scroll.</p>
          </div>
        </div>
      </Layout>
    );
  }

  const content = scroll.content || null;
  const progress = scroll.progress || {};
  const stages = scroll.stages || { scanned: true };
  const readings = scroll.readings || null;
  const mesh = scroll.mesh || null;
  const photo = scroll.photo || null;

  const canonical = `${SITE}/data_browser/${scroll.id}`;
  const metaDesc = metaDescription(scroll);
  // Identify scrolls by their PHerc inventory name; keep "Scroll N" as a tag.
  const nick = scroll.display && scroll.display !== scroll.id ? scroll.display : null;
  const pageTitle = `${scroll.id}${nick ? ` (${nick})` : ""} — Vesuvius Challenge`;

  // og:image: scroll photo, else the first readings image if present.
  const firstReading =
    readings && readings.images && readings.images.length
      ? readings.images[0].src
      : null;
  const ogImageRel = photo || firstReading || null;
  const ogImage = ogImageRel
    ? ogImageRel.startsWith("http")
      ? ogImageRel
      : `${SITE}${ogImageRel}`
    : null;

  // Stats panel values (ref lines 169-173).
  const status = progress.textFound || "Scanned";
  const up = progress.unrolledPct;
  const unrolledPctTxt =
    typeof up === "number" && up
      ? ` · ${up}% unrolled`
      : typeof up === "string" && up
      ? ` · unrolled: ${up}`
      : "";
  const segments =
    progress.segments != null ? progress.segments : scroll.n_segments;
  const segmentsTxt = segments != null ? Number(segments).toLocaleString() : "—";

  // Furthest pipeline stage reached (ref line 134).
  const reached = STAGE_KEYS.filter((k) => stages[k]);
  const furthest = STAGE_LABEL[reached[reached.length - 1]] || "Scanned";

  // Scan lines (ref line 102).
  const scans = scroll.scans || [];

  // schema.org structured data for this scroll (CreativeWork/Dataset).
  const jsonLd = {
    "@context": "https://schema.org",
    "@type": "Dataset",
    name: scroll.id,
    alternateName: scroll.display,
    description: metaDesc,
    url: canonical,
    license: LICENSE_URL,
    isPartOf: {
      "@type": "CreativeWork",
      name: "Vesuvius Challenge",
      url: SITE,
    },
    ...(ogImage ? { image: ogImage } : {}),
    ...(scroll.repository ? { provider: { "@type": "Organization", name: scroll.repository } } : {}),
  };

  return (
    <Layout title={pageTitle} description={metaDesc}>
      <Head>
        <meta name="description" content={metaDesc} />
        <link rel="canonical" href={canonical} />
        <meta property="og:title" content={pageTitle} />
        <meta property="og:description" content={metaDesc} />
        {ogImage ? <meta property="og:image" content={ogImage} /> : null}
      </Head>
      <JsonLd data={jsonLd} />

      <div className="atlas atlas-detail">
        <div className="wrap">
          <Link className="back" to="/data_browser">
            ← all scrolls
          </Link>
          <h1>
            {scroll.id}
            {nick ? <span className="nick">{nick}</span> : null}
            <span
              className={`type ${scroll.type === "scroll" ? "" : "fragment"}`}
            >
              {scroll.type || "sample"}
            </span>
          </h1>
          <div className="repo">📍 {scroll.repository || ""}</div>

          <div className="grid">
            {/* Recovered-text panel */}
            {content && content.summary ? (
              <div className="panel full textpanel">
                <h2>Recovered text</h2>
                <div className="work">
                  {content.work || "Not yet identified"}
                  {content.lang ? (
                    <span className="lang"> · {content.lang}</span>
                  ) : null}
                </div>
                <p className="txt">{stripMarkdown(content.summary)}</p>
                {content.caveat ? (
                  <p className="caveat">⚠️ {stripMarkdown(content.caveat)}</p>
                ) : null}
              </div>
            ) : null}

            {/* 3D model / fragment photo panel */}
            {mesh ? (
              <div className="panel">
                <h2>3D model</h2>
                <BrowserOnly
                  fallback={
                    <div className="atlas-view">
                      <div className="ph">loading…</div>
                    </div>
                  }
                >
                  {() => <ScrollViewer mesh={mesh} />}
                </BrowserOnly>
              </div>
            ) : photo ? (
              <div className="panel">
                <h2>Fragment</h2>
                <div className="fragview">
                  <img
                    src={photo}
                    alt={`photo of ${scroll.id}`}
                    loading="lazy"
                  />
                </div>
              </div>
            ) : (
              <div className="panel">
                <h2>3D model</h2>
                <div className="atlas-view">
                  <div className="ph">no 3D model available</div>
                </div>
              </div>
            )}

            {/* Stats panel */}
            <div className="panel">
              {mesh && photo ? (
                <div className="photo">
                  <img
                    src={photo}
                    alt={`photo of ${scroll.id}`}
                    loading="lazy"
                  />
                </div>
              ) : null}
              <h2 style={mesh && photo ? { marginTop: "14px" } : undefined}>
                Stats
              </h2>
              <dl>
                <dt>Status</dt>
                <dd>
                  <b style={{ color: "var(--accent)" }}>{status}</b>
                  {unrolledPctTxt}
                </dd>
                <dt>Segments</dt>
                <dd>
                  {segmentsTxt}
                  {progress.patches ? (
                    <span style={{ color: "var(--dim)", fontSize: "12px" }}>
                      {" "}
                      / {Number(progress.patches).toLocaleString()} patches
                    </span>
                  ) : null}
                </dd>
                <dt>Min pixel size</dt>
                <dd>{scroll.min_px ? `${scroll.min_px} µm` : "—"}</dd>
                <dt>Scans</dt>
                <dd>{scroll.n_scans}</dd>
                <dt>Volumes</dt>
                <dd>{scroll.n_volumes}</dd>
              </dl>

              {/* Pipeline stepper */}
              <div
                className="steps2"
                role="img"
                aria-label="pipeline progress"
              >
                {STAGE_KEYS.map((k, i) => (
                  <span
                    key={k}
                    className={stages[k] ? `f${i + 1}` : ""}
                    title={`${STAGE_LABEL[k]}${stages[k] ? " ✓" : ""}`}
                  />
                ))}
              </div>
              <div className="plab2">
                Reached: <b style={{ color: "var(--ink)" }}>{furthest}</b>
              </div>

              {/* Scans list */}
              <div className="scans">
                <b style={{ color: "var(--dim)", fontSize: "12px" }}>SCANS</b>
                {scans.length ? (
                  scans.map((sc, i) => {
                    const label =
                      sc.name ||
                      `${sc.px ? `${sc.px}µm ` : ""}${
                        sc.energy ? `${sc.energy}keV ` : ""
                      }${sc.loc || ""}`.trim();
                    return (
                      <div className="scan" key={i}>
                        <span>{label || "—"}</span>
                      </div>
                    );
                  })
                ) : (
                  <div className="scan">
                    <span>—</span>
                  </div>
                )}
              </div>
            </div>

            {/* About panel */}
            {scroll.note ? (
              <div className="panel">
                <h2>About this scroll</h2>
                <p className="txt">{stripMarkdown(scroll.note)}</p>
              </div>
            ) : null}

            {/* Data & access */}
            <DataCatalog scroll={scroll} />

            {/* Ink detection & renders */}
            <ReadingsGallery readings={readings} display={scroll.display} />

            {/* Per-segment ink-detection predictions (from metadata.json) */}
            <InkSegmentsGallery
              segments={scroll.inkSegments}
              display={scroll.display}
            />

            {/* Footer */}
            <div className="footer">
              {scroll._general || ""} &nbsp;·&nbsp;{" "}
              <a
                href="https://dl.ash2txt.org/"
                target="_blank"
                rel="noopener noreferrer"
              >
                ash2txt data browser
              </a>{" "}
              ·{" "}
              <a
                href="/data"
                target="_blank"
                rel="noopener noreferrer"
              >
                scrollprize.org/data
              </a>
            </div>
          </div>
        </div>
      </div>
    </Layout>
  );
}
