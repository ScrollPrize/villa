/**
 * buildIndex — the PURE derivation shared by the build-time generator
 * (scripts/genAtlasData.js) and the runtime data browser (AtlasBrowser.js /
 * ScrollDetailPage.js). Given the public S3 `samples` map plus the curated
 * overlay, it produces the exact `{updated,_general,dashboard,timeline,scrolls}`
 * shape the browser consumes.
 *
 * IMPORTANT: this module is imported by BOTH Node (CommonJS, via the build
 * script) and webpack (the React app). Keep it framework-agnostic — NO `fs`,
 * `path`, `fetch`, or any Node built-in / browser API. All inputs are passed in
 * by the caller; all outputs are plain data. This is why the two consumers can
 * never drift: they run the same code on the same inputs.
 *
 * EMBARGO: the `readings.embargoed` flag + EMBARGOED set remain the mechanism
 * for withholding recovered-text readings; the Naples set lifted 2026-06-25 so
 * EMBARGOED is currently empty.
 */

// Scrolls whose READINGS are embargoed — never publish their readings. The
// Naples embargo (PHerc1667 + PHerc0139) lifted at the 2026-06-25 reveal, so
// this set is now empty; the mechanism stays for any future embargo.
const EMBARGOED = new Set([]);

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

  const scanList = Object.entries(scans).map(([sid, scan]) => {
    const f = scanFacts(scan);
    // Link a scan to the OME-Zarr volume reconstructed from it (→ Neuroglancer).
    const vol = Object.values(volumes).find((v) => v.scan_id === sid);
    let volume = null;
    if (vol) {
      const z = (vol.data || []).find((d) => /zarr/i.test(d.type));
      const o = z && (z.origins || [])[0];
      if (o && o.path) volume = `${((o.access_roots || [])[0] || {}).url || ""}/${o.path}`;
    }
    return { ...f, volume };
  });
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
    const sid = seg.suffix || seg.long_id || key;
    // Per-segment surface layers (→ Neuroglancer) and mesh (→ download).
    const oneUrl = (d) => {
      const o = d && (d.origins || [])[0];
      return o && o.path ? `${((o.access_roots || [])[0] || {}).url || ""}/${o.path}` : null;
    };
    // 2D ink-detection: the flattened prediction image + a downsampled preview.
    const fulls = toUrls(data.filter((d) => d.type === "ink-detection"));
    const downs = toUrls(data.filter((d) => d.type === "ink-detection-downsampled"));
    const full = fulls.length ? fulls.sort(byUm)[0] : null;
    const down = downs.sort(byUm)[0] || null;
    // 3D ink (alpha-render): ink painted on the rendered surface. The full .tif
    // is tens of MB (download only); the ds8 .jpg is the Thumbor thumbnail src.
    // s3ToHttp so the jpg matches the thumbnail prefix and the tif is clickable.
    const alphaEntry = data.find((d) => d.type === "alpha-render");
    const alphaDsEntry = data.find((d) => d.type === "alpha-render-downsampled");
    const alpha = alphaEntry ? s3ToHttp(oneUrl(alphaEntry)) : null;
    const alphaPrev = alphaDsEntry ? s3ToHttp(oneUrl(alphaDsEntry)) : null;
    // Emit a segment if it has EITHER a 2D ink prediction OR a 3D-ink preview
    // (so the 3 alpha-only PHercParis4 segments still appear).
    if (!full && !alphaPrev) continue;
    const layers = oneUrl(data.find((d) => d.type === "layers-zarr"));
    const mesh = oneUrl(data.find((d) => d.type === "obj-flattened" || d.type === "obj"));
    out.push({
      id: key,
      label: inkSegmentLabel(sid),
      um: full ? full.um : null,
      full: full ? full.url : null,
      down: down ? down.url : null,
      preview: full ? (down ? down.url : full.url) : null,
      alpha,
      alphaPrev,
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

// Volume-level model predictions for a sample (surface-prediction / 3D-ink) —
// the "Predictions" table the old atlas exposed. A prediction lives in the same
// volume object as the base CT ome-zarr, so the volume's properties give the
// base resolution/energy and the volume map key IS the base-volume id. The raw
// s3:// zarr URL is kept as-is; the component builds Neuroglancer/Files links
// via neuroglancerUrl()/toHttp(), exactly like scans[].volume.
function extractPredictions(sample) {
  const vols = (sample && sample.volumes) || {};
  const oneUrl = (d) => {
    const o = d && (d.origins || [])[0];
    return o && o.path ? `${((o.access_roots || [])[0] || {}).url || ""}/${o.path}` : null;
  };
  const out = [];
  for (const [volKey, v] of Object.entries(vols)) {
    const props = v.properties || {};
    // The raw CT this volume reconstructs (same volume object) — the base layer
    // the prediction is overlaid onto in Neuroglancer.
    const baseZarr = oneUrl((v.data || []).find((d) => d.type === "ome-zarr"));
    for (const d of v.data || []) {
      if (d.type !== "surface-prediction-zarr" && d.type !== "ink-detection-3d-zarr")
        continue;
      const p = d.parameters || {};
      out.push({
        purpose: d.type.replace(/-zarr$/, ""), // surface-prediction | ink-detection-3d
        baseVolume: volKey,
        px: props.pixel_size_um ?? null,
        energy: props.energy_keV ?? null,
        model: p.model_id ?? null,
        level: p.level ?? null,
        threshold: p.threshold_value ?? null,
        zarr: oneUrl(d),
        baseZarr,
      });
    }
  }
  // 3D-ink first (the showcase), then surface predictions by finest pixel size,
  // then model id — a stable, meaningful order for the table.
  const rank = (x) => (x.purpose === "ink-detection-3d" ? 0 : 1);
  out.sort((a, b) => {
    if (rank(a) !== rank(b)) return rank(a) - rank(b);
    const pa = a.px ?? 1e9;
    const pb = b.px ?? 1e9;
    if (pa !== pb) return pa - pb;
    return String(a.model || "").localeCompare(String(b.model || ""));
  });
  // Defensive de-dup on the identifying tuple.
  const seen = new Set();
  return out.filter((x) => {
    const k = `${x.purpose}|${x.baseVolume}|${x.model}|${x.level}|${x.threshold}`;
    if (seen.has(k)) return false;
    seen.add(k);
    return true;
  });
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
    // Per-scroll legal/access terms (e.g. the PHerc. Paris 4 publication
    // reservation) rendered as a notice panel on the detail page.
    legalNotice: o.legalNotice ?? null,
    // License name -> which of this scroll's data that license covers
    // (annotates the License row in the Data & access panel).
    licenseScope: o.licenseScope ?? null,
    // Curated license list override — for items whose S3 metadata carries no
    // per-volume license (e.g. the legacy DLS fragment volpkgs, which are
    // EduceLab-licensed but volume-less in metadata.json).
    ...(o.licenses ? { licenses: o.licenses } : {}),
    // DEPRECATED: surface-prediction Neuroglancer links now derive from
    // scroll.predictions (see extractPredictions); kept for backward-compat only.
    volume: o.volume ?? null,
  };
}

// Derive the count-based dashboard numbers from the per-scroll data so they can
// never drift out of sync with the pipeline `stages` or the scan metadata.
// Editorial fields (the PB headline, Latin subs, labels, descriptions) are left
// exactly as curated. Tiles opt in via a `derive` key: "all" | "ink" | "text"
// set the big number; "resolution" sets the scan-resolution sub-label.
function deriveDashboard(dashboard, scrolls) {
  if (!dashboard) return dashboard;
  // Clone before mutating: `dashboard` is the caller's overlay.dashboard, which
  // at runtime is a shared webpack module singleton reused across every
  // buildIndex() call (grid + detail page, filtered + unfiltered scroll sets).
  // Mutating it in place would leak counts between consumers; deep-copy so this
  // function stays pure and each call derives independently.
  dashboard = structuredClone(dashboard);
  const cnt = (k) => scrolls.filter((s) => s.stages && s.stages[k]).length;
  // Scan-resolution facts across every sample's scans (for the data tile).
  const scanPx = scrolls
    .flatMap((s) => (s.scans || []).map((x) => x.px))
    .filter((v) => v != null);
  const subCount = scanPx.filter((v) => v < 2).length;
  const finest = scanPx.length ? Math.min(...scanPx) : null;
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
  if (dashboard.totals) {
    for (const k of ["all", "scrolls", "fragments", "segmented", "unrolled", "ink", "text"])
      dashboard.totals[k] = counts[k];
    dashboard.totals.subum = subCount;
  }
  for (const t of dashboard.tiles || []) {
    if (t.derive === "all") {
      t.big = String(counts.all);
      t.sub = `${counts.scrolls} scrolls · ${counts.fragments} fragments`;
    } else if (t.derive === "resolution") {
      if (finest != null)
        t.sub = `${subCount} scan${subCount === 1 ? "" : "s"} at sub-2µm, down to ${finest.toFixed(1)}µm`;
    } else if (t.derive && counts[t.derive] !== undefined) {
      t.big = String(counts[t.derive]);
    }
  }
  return dashboard;
}

// Furthest pipeline stage reached, as a 0..4 rank (text highest) — the primary
// grid order, so recovered scrolls rank above less-progressed ones regardless
// of their (sometimes stale) progress.score.
const STAGE_ORDER = ["scanned", "segmented", "unrolled", "ink", "text"];
function stageRank(s) {
  const st = (s && s.stages) || {};
  let r = 0;
  STAGE_ORDER.forEach((k, i) => { if (st[k]) r = i; });
  return r;
}

/**
 * Merge the public S3 `samples` map with the curated overlay into the browser's
 * `index.json` shape. `samples` may be empty (overlay-only fallback). `opts`:
 *   - overlay:      parsed src/data/atlasOverlay.json ({scrolls,dashboard,...})
 *   - meshManifest: parsed static/img/data_browser/meshes/manifest.json ({id:mesh})
 *   - rewrites:     dataAccess.rewrites ([{from,to}]) for s3://→https thumbnails
 */
function buildIndex(samples, opts) {
  const { overlay = {}, meshManifest = {}, rewrites = [] } = opts || {};
  const overlayScrolls = overlay.scrolls || {};
  samples = samples || {};

  const s3ToHttp = (url) => {
    for (const r of rewrites) if (url.startsWith(r.from)) return r.to + url.slice(r.from.length);
    return url;
  };

  const ids = uniqStrings([
    ...Object.keys(samples),
    ...Object.keys(overlayScrolls),
  ]);

  const scrolls = ids.map((id) => {
    const facts = samples[id] ? deriveFacts(samples[id]) : null;
    const curated = curatedFields(overlayScrolls[id], id);
    const o = overlayScrolls[id] || {};
    const mesh = meshManifest[id] || null;
    const inkSegments = samples[id]
      ? extractInkSegments(samples[id], id, s3ToHttp)
      : [];
    const predictions = samples[id] ? extractPredictions(samples[id]) : [];

    // Prefer S3 for factual fields; fall back to nothing for overlay-only ids
    // without S3 facts (still emit with what's available).
    return {
      id,
      // factual (S3) — null fields where S3 has no record for this id; a curated
      // overlay `type`/`desc` takes precedence when present.
      type: o.type || (facts ? facts.type : "scroll"),
      desc: o.desc || (facts ? facts.desc : ""),
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
      // predictions (volume-level) + derived flags/counts for cards & filters
      predictions,
      hasSurfacePred: predictions.some((p) => p.purpose === "surface-prediction"),
      hasInk3d: predictions.some((p) => p.purpose === "ink-detection-3d"),
      n_predictions: predictions.length,
      n_inkSegments: inkSegments.filter((s) => s.full).length,
      n_alpha: inkSegments.filter((s) => s.alphaPrev).length,
      alphaHero: (inkSegments.find((s) => s.alphaPrev) || {}).alphaPrev || null,
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

  return {
    updated: (overlay.dashboard && overlay.dashboard.updated) || null,
    _general: overlay._general || "",
    dashboard: deriveDashboard(overlay.dashboard || null, scrolls),
    timeline: overlay.timeline || [],
    scrolls,
  };
}

module.exports = { buildIndex, EMBARGOED, stageRank };
