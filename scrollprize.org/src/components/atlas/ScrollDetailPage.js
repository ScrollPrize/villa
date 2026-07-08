import React from "react";
import Layout from "@theme/Layout";
import Head from "@docusaurus/Head";
import Link from "@docusaurus/Link";
import BrowserOnly from "@docusaurus/BrowserOnly";
import JsonLd from "@site/src/components/JsonLd";
import ScrollViewer from "./ScrollViewer";
import DataCatalog from "./DataCatalog";
import PredictionsPanel from "./PredictionsPanel";
import { photoThumb, neuroglancerUrl } from "./dataAccess";
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

// Pipeline stepper config. 4 stages — segmentation and unrolling are merged
// into one "Segmented" stage (unrolling is a subset of segmentation); per-scroll
// unrolling progress is shown separately as "% unrolled". Uses the `.steps2` /
// `.plab2` classes (distinct from the index card's `.steps`).
const STAGES = [
  { key: "scanned", label: "Scanned", cls: "f1" },
  { key: "segmented", label: "Segmenting", cls: "f2" },
  { key: "ink", label: "Ink detected", cls: "f4" },
  { key: "text", label: "Text recovered", cls: "f5" },
];
const stageReached = (stages, key) =>
  key === "segmented" ? !!(stages.segmented || stages.unrolled) : !!stages[key];

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
  const label = scroll.label || scroll.id;
  const nick = scroll.display && scroll.display !== scroll.id ? scroll.display : null;
  const pageTitle = `${label}${nick ? ` (${nick})` : ""} — Vesuvius Challenge`;

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

  // Furthest pipeline stage reached.
  const reachedStages = STAGES.filter((s) => stageReached(stages, s.key));
  const furthest = (reachedStages[reachedStages.length - 1] || STAGES[0]).label;

  // Scan lines (ref line 102).
  const scans = scroll.scans || [];

  // schema.org structured data for this scroll (CreativeWork/Dataset).
  const jsonLd = {
    "@context": "https://schema.org",
    "@type": "Dataset",
    name: label,
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
    ...(scroll.legalNotice
      ? { usageInfo: `${canonical}#publication-notice` }
      : {}),
  };
  const legalNotice = scroll.legalNotice || null;

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
            {label}
            {nick ? <span className="nick">{nick}</span> : null}
            <span
              className={`type ${scroll.type === "scroll" ? "" : "fragment"}`}
            >
              {scroll.type || "sample"}
            </span>
          </h1>
          <div className="repo">📍 {scroll.repository || ""}</div>

          <div className="grid">
            {/* Legal/access terms (e.g. the publication-reservation period
                that conditioned the imaging of this scroll). Rendered first —
                readers must see it before any data link. */}
            {legalNotice ? (
              <div
                className="panel full"
                id="publication-notice"
                style={{ borderColor: "var(--accent)" }}
              >
                <h2>⚖️ {legalNotice.title || "Data access notice"}</h2>
                {(legalNotice.paragraphs || []).map((p, i) => (
                  <p className="txt" key={i}>
                    {p}
                  </p>
                ))}
                {legalNotice.licenseLine ? (
                  <p
                    className="txt"
                    style={{ color: "var(--dim)", fontSize: "13px" }}
                  >
                    {legalNotice.licenseLine}
                  </p>
                ) : null}
              </div>
            ) : null}

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

            {/* Hero: the physical photograph if available, else the 3D model */}
            {photo ? (
              <div className="panel">
                <h2>Photograph</h2>
                <div className="fragview">
                  <img
                    src={photoThumb(photo, 1400)}
                    alt={`Photograph of ${label}`}
                    loading="lazy"
                    onError={(e) => {
                      e.currentTarget.src = photo;
                    }}
                  />
                </div>
              </div>
            ) : mesh ? (
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
            ) : (
              <div className="panel">
                <h2>3D model</h2>
                <div className="atlas-view">
                  <div className="ph">no 3D model available</div>
                </div>
              </div>
            )}

            {/* Secondary 3D-model panel when the item also has a mesh */}
            {photo && mesh ? (
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
            ) : null}

            {/* Stats panel */}
            <div className="panel">
              <h2>Stats</h2>
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
                {STAGES.map((s) => {
                  const on = stageReached(stages, s.key);
                  return (
                    <span
                      key={s.key}
                      className={on ? s.cls : ""}
                      title={`${s.label}${on ? " ✓" : ""}`}
                    />
                  );
                })}
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
                        {sc.volume ? (
                          <a
                            href={neuroglancerUrl(sc.volume, label)}
                            target="_blank"
                            rel="noopener noreferrer"
                          >
                            {label || "—"} ↗
                          </a>
                        ) : (
                          <span>{label || "—"}</span>
                        )}
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
            {scroll.note || scroll.desc ? (
              <div className="panel">
                <h2>About this {scroll.type === "fragment" ? "fragment" : "scroll"}</h2>
                {scroll.note ? (
                  <p className="txt">{stripMarkdown(scroll.note)}</p>
                ) : null}
                {scroll.desc ? (
                  <p className="txt">{stripMarkdown(scroll.desc)}</p>
                ) : null}
              </div>
            ) : null}

            {/* Data & access */}
            <DataCatalog scroll={scroll} />

            {/* Model predictions (volume-level surface + 3D-ink) */}
            <PredictionsPanel predictions={scroll.predictions} />

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
