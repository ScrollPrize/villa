#!/usr/bin/env node
/**
 * Build-time data generator for the /data_browser page.
 *
 * Produces a hybrid, PUBLIC-ONLY dataset by merging two sources:
 *   1. FACTUAL fields regenerated live from the public S3 metadata.json
 *      (https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/metadata.json)
 *      — scan/volume/segment counts, pixel sizes, energies, scan locations, etc.
 *   2. A human-editable CURATED overlay checked into the repo
 *      (src/data/atlasOverlay.json) — display names, notes, readings, content,
 *      progress, stages, plus the dashboard / timeline / general copy.
 *
 * EMBARGO: PHerc1667 and PHerc0139 had recovered-text/ink READINGS embargoed
 * until the Naples reveal (2026-06-25); that embargo has lifted, so they are
 * now published like any other scroll. The EMBARGOED set + `readings.embargoed`
 * flag remain in place as the mechanism for any future embargo.
 *
 * Output: static/data_browser/index.json
 *
 * Run directly (Node 20+ required for global fetch / AbortController):
 *   node scripts/genAtlasData.js
 *
 * Mirrors the style of scripts/genImageDimensions.js and
 * scripts/fetchLatestPosts.js — CommonJS, no extra npm deps.
 */
const fs = require("fs");
const path = require("path");

const ROOT = path.resolve(__dirname, "..");
const OVERLAY_PATH = path.join(ROOT, "src", "data", "atlasOverlay.json");
const MESH_MANIFEST_PATH = path.join(
  ROOT,
  "static",
  "img",
  "data_browser",
  "meshes",
  "manifest.json"
);
const OUT_DIR = path.join(ROOT, "static", "data_browser");
const OUT_PATH = path.join(OUT_DIR, "index.json");
const DATA_ACCESS_PATH = path.join(ROOT, "src", "data", "atlasDataAccess.json");

const S3_METADATA_URL =
  "https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/metadata.json";
const FETCH_TIMEOUT_MS = 25000;

// Scrolls whose READINGS are embargoed — never publish their readings. The
// Naples embargo (PHerc1667 + PHerc0139) lifted at the 2026-06-25 reveal, so
// this set is now empty; the mechanism stays for any future embargo.
const EMBARGOED = new Set([]);

function readJsonIfExists(p) {
  try {
    if (fs.existsSync(p)) return JSON.parse(fs.readFileSync(p, "utf8"));
  } catch (e) {
    console.warn(`[atlas] failed to read ${path.relative(ROOT, p)}: ${e.message}`);
  }
  return null;
}

async function fetchMetadata() {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), FETCH_TIMEOUT_MS);
  try {
    const res = await fetch(S3_METADATA_URL, { signal: controller.signal });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return await res.json();
  } finally {
    clearTimeout(timer);
  }
}

function uniqSortedNums(arr) {
  return [...new Set(arr.filter((v) => v !== null && v !== undefined))].sort(
    (a, b) => a - b
  );
}
function uniqStrings(arr) {
  return [...new Set(arr.filter((v) => v !== null && v !== undefined && v !== ""))];
}

// Pull {px, energy, loc, name} out of a single scan record. The public
// metadata stores location only under `creation.metadata`, while energy /
// pixel size live both there and (sometimes) under `properties`. Be tolerant.
function scanFacts(scan) {
  const cm = (scan && scan.creation && scan.creation.metadata) || {};
  const props = (scan && scan.properties) || {};
  const px = cm.pixel_size_um ?? props.pixel_size_um ?? null;
  const energy = cm.energy_keV ?? props.energy_keV ?? null;
  const loc = cm.location ?? props.location ?? null;
  const name = (scan && scan.long_id) || null;
  return { px, energy, loc, name };
}

// Derive the FACTUAL fields for one sample from its S3 record.
function deriveFacts(sample) {
  const sampleProps = (sample.sample && sample.sample.properties) || {};
  const scans = sample.scans || {};
  const volumes = sample.volumes || {};
  const segments = sample.segments || {};

  const scanList = Object.values(scans).map(scanFacts);
  const pxVals = scanList.map((s) => s.px).filter((v) => v !== null && v !== undefined);

  // Distinct data licenses across the item's volumes (varies: EduceLab vs
  // CC BY-NC), plus the first OME-Zarr CT volume for a Neuroglancer link.
  const licenses = [];
  const seenLic = new Set();
  let volumeZarr = null;
  for (const v of Object.values(volumes)) {
    const lic = v.properties && v.properties.license;
    if (lic && lic.url && !seenLic.has(lic.url)) {
      seenLic.add(lic.url);
      licenses.push({ name: lic.name || "License", url: lic.url });
    }
    if (!volumeZarr) {
      const z = (v.data || []).find((d) => /zarr/i.test(d.type));
      const o = z && (z.origins || [])[0];
      if (o && o.path) volumeZarr = `${((o.access_roots || [])[0] || {}).url || ""}/${o.path}`;
    }
  }

  return {
    // Samples with no explicit type in the metadata are scrolls (per curation).
    type: sampleProps.type && sampleProps.type !== "?" ? sampleProps.type : "scroll",
    desc: (sample.sample && sample.sample.description) || "",
    legacy: sampleProps.legacy_data_url ?? null,
    n_scans: Object.keys(scans).length,
    n_volumes: Object.keys(volumes).length,
    n_segments: Object.keys(segments).length,
    min_px: pxVals.length ? Math.min(...pxVals) : null,
    energies: uniqSortedNums(scanList.map((s) => s.energy)),
    locations: uniqStrings(scanList.map((s) => s.loc)),
    scans: scanList,
    licenses,
    volumeZarr,
  };
}

// Parse a resolution like "2.4um" / "7.91um" out of an ink-detection filename.
function umFromName(name) {
  const m = String(name || "").match(/(\d+(?:\.\d+)?)um/);
  return m ? parseFloat(m[1]) : null;
}

// Friendly segment label: prefer the winding tag (w052 / w053-058), then the
// auto-grown index, then a bare 14-digit timestamp id. A trailing variant
// qualifier (jordi / v14 / v2 …) is appended so distinct segments covering the
// same windings don't collapse to an identical label.
function inkSegmentLabel(raw) {
  const s = String(raw || "");
  const ag = s.match(/auto_grown_\d+_(\d+)/i);
  if (ag) return "auto-grown " + ag[1];
  const w = s.match(/w(\d+(?:-\d+)?)/i);
  const ts = s.match(/^(\d{14})/);
  const base = w ? "w" + w[1] : ts ? ts[1] : s;
  const v = s.match(/_(jordi|v\d+)/i);
  return v ? `${base} · ${v[1]}` : base;
}

// Per-segment ink-detection predictions for a sample, as public HTTPS links —
// the per-segment list the old atlas exposed. One entry per segment (its
// finest-resolution run) with a downsampled preview for thumbnailing. Embargoed
// scrolls expose nothing. `s3ToHttp` maps a segment's s3:// access-root to a
// public https URL via the dataAccess rewrites.
function extractInkSegments(sample, id, s3ToHttp) {
  if (EMBARGOED.has(id)) return [];
  const segs = (sample && sample.segments) || {};
  const toUrls = (entries) =>
    entries.map((d) => {
      const o = (d.origins || [])[0] || {};
      const root = ((o.access_roots || [])[0] || {}).url || "";
      const p = o.path || "";
      return { url: s3ToHttp(`${root}/${p}`), um: umFromName(p) };
    });
  const byUm = (a, b) => (a.um ?? 1e9) - (b.um ?? 1e9);
  const out = [];
  for (const key of Object.keys(segs)) {
    const seg = segs[key];
    const data = seg.data || [];
    const fulls = toUrls(data.filter((d) => d.type === "ink-detection"));
    if (!fulls.length) continue;
    const downs = toUrls(data.filter((d) => d.type === "ink-detection-downsampled"));
    const full = fulls.sort(byUm)[0];
    const down = downs.sort(byUm)[0] || null;
    const sid = seg.suffix || seg.long_id || key;
    // Per-segment surface layers (→ Neuroglancer) and mesh (→ download).
    const oneUrl = (d) => {
      const o = d && (d.origins || [])[0];
      return o && o.path ? `${((o.access_roots || [])[0] || {}).url || ""}/${o.path}` : null;
    };
    const layers = oneUrl(data.find((d) => d.type === "layers-zarr"));
    const mesh = oneUrl(data.find((d) => d.type === "obj-flattened" || d.type === "obj"));
    out.push({
      id: key,
      label: inkSegmentLabel(sid),
      um: full.um,
      full: full.url,
      preview: down ? down.url : full.url,
      layers,
      mesh,
    });
  }
  // Order unwrap (winding) ranges in DECREASING order (highest winding first);
  // segments without a winding number (auto-grown, legacy timestamp segments)
  // sort to the end.
  const wnum = (s) => {
    const m = String(s.label).match(/^w(\d+)/i);
    return m ? parseInt(m[1], 10) : null;
  };
  out.sort((a, b) => {
    const wa = wnum(a);
    const wb = wnum(b);
    if (wa !== null && wb !== null) {
      if (wa !== wb) return wb - wa;
      return String(a.label).localeCompare(String(b.label), undefined, { numeric: true });
    }
    if (wa !== null) return -1;
    if (wb !== null) return 1;
    return String(a.label).localeCompare(String(b.label), undefined, { numeric: true });
  });
  return out;
}

// Merge curated overlay fields onto a scroll, applying the embargo strip.
// A scroll that reached ink/text must also be segmented (and scanned), and text
// implies ink — enforce that chain so the stepper never shows ink/text without
// the prerequisite stages. `unrolled` stays independent: flat fragments get ink
// without unrolling. Operates on a copy; recomputes the stored `furthest`.
function normalizeStages(stages) {
  if (!stages) return stages;
  if (stages.text) stages.ink = true;
  if (stages.ink) stages.segmented = true;
  if (stages.segmented || stages.unrolled || stages.ink || stages.text) stages.scanned = true;
  const ORDER = ["scanned", "segmented", "unrolled", "ink", "text"];
  let f = 0;
  ORDER.forEach((k, i) => { if (stages[k]) f = i; });
  stages.furthest = f;
  return stages;
}

function curatedFields(overlayScroll, id) {
  const o = overlayScroll || {};
  let readings = o.readings ?? null;
  // Embargo: never publish readings for embargoed scrolls (or when the overlay
  // flags them, or when readings are missing for an embargoed id).
  if (EMBARGOED.has(id) || (readings && readings.embargoed === true)) {
    readings = null;
  }
  return {
    display: o.display ?? id,
    repository: o.repository ?? null,
    note: o.note ?? "",
    photo: o.photo ?? null,
    bucketUrl: o.bucketUrl ?? null,
    content: o.content ?? null,
    progress: o.progress ?? null,
    stages: normalizeStages(o.stages ? { ...o.stages } : null),
    readings,
    volume: o.volume ?? null,
  };
}

// Derive the count-based dashboard numbers from the per-scroll data so they can
// never drift out of sync with the pipeline `stages`. Editorial fields (the PB
// tile, the µm/Latin subs, labels, descriptions, totals.at1um/subum) are left
// exactly as curated. Tiles opt in via a `derive` key ("all" | "ink" | "text").
function deriveDashboard(dashboard, scrolls) {
  if (!dashboard) return dashboard;
  const cnt = (k) => scrolls.filter((s) => s.stages && s.stages[k]).length;
  const counts = {
    all: scrolls.length,
    scrolls: scrolls.filter((s) => s.type === "scroll").length,
    fragments: scrolls.filter((s) => s.type !== "scroll").length,
    scanned: cnt("scanned"),
    segmented: cnt("segmented"),
    unrolled: cnt("unrolled"),
    ink: cnt("ink"),
    text: cnt("text"),
  };
  for (const f of dashboard.funnel || [])
    if (counts[f.key] !== undefined) f.count = counts[f.key];
  if (dashboard.totals)
    for (const k of ["all", "scrolls", "fragments", "segmented", "unrolled", "ink", "text"])
      dashboard.totals[k] = counts[k];
  for (const t of dashboard.tiles || []) {
    if (t.derive === "all") {
      t.big = String(counts.all);
      t.sub = `${counts.scrolls} scrolls · ${counts.fragments} fragments`;
    } else if (t.derive && counts[t.derive] !== undefined) {
      t.big = String(counts[t.derive]);
    }
  }
  return dashboard;
}

function generate() {
  const overlay = readJsonIfExists(OVERLAY_PATH) || {};
  const overlayScrolls = overlay.scrolls || {};
  const meshManifest = readJsonIfExists(MESH_MANIFEST_PATH) || {};
  const rewrites = ((readJsonIfExists(DATA_ACCESS_PATH) || {}).dataAccess || {}).rewrites || [];
  const s3ToHttp = (url) => {
    for (const r of rewrites) if (url.startsWith(r.from)) return r.to + url.slice(r.from.length);
    return url;
  };
  // Furthest pipeline stage reached, as a 0..4 rank (text highest) — the primary
  // grid order, so recovered scrolls rank above less-progressed ones regardless
  // of their (sometimes stale) progress.score.
  const STAGE_ORDER = ["scanned", "segmented", "unrolled", "ink", "text"];
  const stageRank = (s) => {
    const st = (s && s.stages) || {};
    let r = 0;
    STAGE_ORDER.forEach((k, i) => { if (st[k]) r = i; });
    return r;
  };

  return fetchMetadata()
    .then((meta) => {
      const samples = (meta && meta.samples) || {};
      const ids = uniqStrings([
        ...Object.keys(samples),
        ...Object.keys(overlayScrolls),
      ]);

      const scrolls = ids.map((id) => {
        const facts = samples[id] ? deriveFacts(samples[id]) : null;
        const curated = curatedFields(overlayScrolls[id], id);
        const mesh = meshManifest[id] || null;
        const inkSegments = samples[id]
          ? extractInkSegments(samples[id], id, s3ToHttp)
          : [];

        // Prefer S3 for factual fields; fall back to nothing for overlay-only
        // ids without S3 facts (still emit with what's available).
        return {
          id,
          // factual (S3) — null fields where S3 has no record for this id
          type: facts ? facts.type : "scroll",
          desc: facts ? facts.desc : "",
          legacy: facts ? facts.legacy : null,
          n_scans: facts ? facts.n_scans : 0,
          n_volumes: facts ? facts.n_volumes : 0,
          n_segments: facts ? facts.n_segments : 0,
          min_px: facts ? facts.min_px : null,
          energies: facts ? facts.energies : [],
          locations: facts ? facts.locations : [],
          scans: facts ? facts.scans : [],
          hasS3: !!facts,
          licenses: facts ? facts.licenses : [],
          volumeZarr: facts ? facts.volumeZarr : null,
          // curated (overlay)
          ...curated,
          // assets
          mesh,
          inkSegments,
        };
      });

      // Sort by furthest pipeline stage (text first), then progress.score desc.
      scrolls.sort((a, b) => {
        const ra = stageRank(a);
        const rb = stageRank(b);
        if (ra !== rb) return rb - ra;
        const sa = (a.progress && typeof a.progress.score === "number" && a.progress.score) || 0;
        const sb = (b.progress && typeof b.progress.score === "number" && b.progress.score) || 0;
        return sb - sa;
      });

      const out = {
        updated: (overlay.dashboard && overlay.dashboard.updated) || null,
        _general: overlay._general || "",
        dashboard: deriveDashboard(overlay.dashboard || null, scrolls),
        timeline: overlay.timeline || [],
        scrolls,
      };

      fs.mkdirSync(OUT_DIR, { recursive: true });
      fs.writeFileSync(OUT_PATH, JSON.stringify(out, null, 2) + "\n");
      const withMesh = scrolls.filter((s) => s.mesh).length;
      const withReadings = scrolls.filter((s) => s.readings).length;
      const withContent = scrolls.filter((s) => s.content).length;
      console.log(
        `[atlas] wrote ${scrolls.length} scrolls to ${path.relative(ROOT, OUT_PATH)} ` +
          `(${withMesh} mesh, ${withReadings} readings, ${withContent} content)`
      );
    })
    .catch((err) => {
      console.warn(`[atlas] S3 metadata fetch failed: ${err.message}`);
      if (fs.existsSync(OUT_PATH)) {
        console.warn(
          `[atlas] reusing existing ${path.relative(ROOT, OUT_PATH)} (build continues)`
        );
        return;
      }
      console.warn(
        `[atlas] no existing ${path.relative(ROOT, OUT_PATH)} to fall back to; writing overlay-only data`
      );
      // Last resort: emit overlay-only data so the build does not crash.
      const ids = Object.keys(overlayScrolls);
      const scrolls = ids
        .map((id) => ({
          id,
          type: "scroll",
          desc: "",
          legacy: null,
          n_scans: 0,
          n_volumes: 0,
          n_segments: 0,
          min_px: null,
          energies: [],
          locations: [],
          scans: [],
          hasS3: false,
          licenses: [],
          volumeZarr: null,
          ...curatedFields(overlayScrolls[id], id),
          mesh: meshManifest[id] || null,
          inkSegments: [],
        }))
        .sort((a, b) => {
          const ra = stageRank(a);
          const rb = stageRank(b);
          if (ra !== rb) return rb - ra;
          const sa = (a.progress && a.progress.score) || 0;
          const sb = (b.progress && b.progress.score) || 0;
          return sb - sa;
        });
      const out = {
        updated: (overlay.dashboard && overlay.dashboard.updated) || null,
        _general: overlay._general || "",
        dashboard: deriveDashboard(overlay.dashboard || null, scrolls),
        timeline: overlay.timeline || [],
        scrolls,
      };
      fs.mkdirSync(OUT_DIR, { recursive: true });
      fs.writeFileSync(OUT_PATH, JSON.stringify(out, null, 2) + "\n");
      console.log(`[atlas] wrote ${scrolls.length} overlay-only scrolls (no S3 facts)`);
    });
}

module.exports = { generate };

if (require.main === module) {
  generate();
}
