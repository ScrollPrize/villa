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

// Parse a resolution like "2.4um" / "7.91um" out of an ink-detection file
// path. The FILENAME is tried before the full path: a segment FOLDER name can
// itself contain a resolution tag (PHerc0814's "…-46527_2um_try2/…-1.129um-…"
// really exists) and a whole-path match would grab that "2um" for every
// variant — mislabeling the segment and, worse, erasing the um difference the
// live metadata.min.json path (no target_volume → no volume-properties um)
// relies on to order/label its variants.
function umFromName(name) {
  const s = String(name || "");
  const m = (s.split("/").pop() || "").match(/(\d+(?:\.\d+)?)um/) || s.match(/(\d+(?:\.\d+)?)um/);
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

// Resolve one full-resolution render `data` entry (ink-detection / alpha-render)
// to its public URL plus joined per-variant metadata. um/energy come
// AUTHORITATIVELY from the target volume's own properties (pixel_size_um /
// energy_keV) when this sample's `volumes` map carries that volume; um falls
// back to parsing the filename (umFromName) for the should-not-happen case of
// a target_volume the sample doesn't list — energy has no such fallback (no
// filename signal to parse it from) and simply stays null.
function toRenderMeta(d, s3ToHttp, volumes) {
  const o = (d.origins || [])[0] || {};
  const root = ((o.access_roots || [])[0] || {}).url || "";
  const p = o.path || "";
  const targetVolume = d.parameters?.target_volume ?? null;
  const volProps =
    (targetVolume != null && volumes[targetVolume] && volumes[targetVolume].properties) || null;
  return {
    url: s3ToHttp(`${root}/${p}`),
    um: volProps?.pixel_size_um ?? umFromName(p),
    energy: volProps?.energy_keV ?? null,
    targetVolume,
    modelId: d.parameters?.model_id ?? null,
    modelName: d.creation_info?.provenance?.parameters?.model ?? null,
    renderLevel: d.parameters?.render_level ?? null,
  };
}

// Collapse a list of toRenderMeta() entries to (at most) one per target
// volume. Entries with no target_volume (data that predates the field) each
// keep their own bucket — they must never collide into a shared "null volume"
// bucket. The one genuine same-volume collision seen in the wild is duplicate
// model runs (e.g. PHerc0172 segment 20251107110950: 2 target_volumes x 2
// model_ids each) — model ids are timestamp strings, so keeping the
// lexicographically largest one picks the newest run, deterministically.
// Returns the surviving one-per-volume list sorted finest um first.
function groupRendersByVolume(entries) {
  const buckets = new Map();
  let anon = 0;
  for (const e of entries) {
    const key = e.targetVolume != null ? `v:${e.targetVolume}` : `_:${anon++}`;
    const cur = buckets.get(key);
    if (!cur || String(e.modelId || "") > String(cur.modelId || "")) buckets.set(key, e);
  }
  return [...buckets.values()].sort((a, b) => (a.um ?? 1e9) - (b.um ?? 1e9));
}

// Pair one surviving full-resolution variant `g` with its downsampled
// counterpart out of `downs`. target_volume is the authoritative join key —
// BUT the live metadata.min.json projection has been observed (PHerc0139,
// 2026-07) publishing `target_volume: null` on BOTH the full and downsampled
// sides, where the old bare `dw.targetVolume === g.targetVolume` join meant
// null === null: EVERY variant matched the FIRST downsampled entry, so the
// compare view drew the identical image on both layers (and the grid/lightbox
// thumbnail could belong to the other variant). Fall back through the
// filename stem (a downsampled render is "<full stem>-ds8.jpg" by
// convention), then model_id, then the unambiguous lone-pair case; otherwise
// return null — no thumbnail is strictly better than the wrong variant's.
function downFor(g, downs, nVariants) {
  if (g.targetVolume != null) {
    const byVol = downs.find((dw) => dw.targetVolume === g.targetVolume);
    if (byVol) return byVol;
  }
  const stem = String(g.url || "")
    .split("/")
    .pop()
    .replace(/\.[A-Za-z0-9]+$/, "");
  if (stem) {
    const byStem = downs.find((dw) => String(dw.url || "").includes(stem));
    if (byStem) return byStem;
  }
  if (g.modelId != null) {
    const byModel = downs.find((dw) => dw.modelId === g.modelId);
    if (byModel) return byModel;
  }
  return nVariants === 1 && downs.length === 1 ? downs[0] : null;
}

// Reorder multi-variant renders so the variant with the MOST VISIBLE CONTENT
// comes first (and so becomes the segment's primary image everywhere a single
// representative is shown: grid thumbnail, default lightbox view, left side of
// the compare sweep). Finest-um-first turned out to be a poor default: the
// finest volume is typically a partial-coverage high-res rescan, so its render
// is mostly black where the segment leaves the scanned region — measured
// across all 113 multi-variant segments (2026-07), the finest variant was the
// blacker one in 104 of them.
//
// The signal is `renderSizes[thumbUrl] * um²`: the downsampled JPEG's byte
// size normalized to physical papyrus area. Both variants of a segment cover
// the same physical area at the same -ds8 factor, so bytes×um² ∝ bytes/pixel —
// JPEG entropy density — and a mostly-black render compresses far smaller per
// pixel than one full of detected ink. (Raw bytes alone mislead: a finer-um
// render has more pixels, which can out-byte a fuller coarse render — the um²
// normalization picked the measurably-less-black variant in 113/113 segments,
// raw bytes only 102/113.) `renderSizes` ({thumbUrl: bytes}) is fetched
// OUTSIDE this pure module (scripts/genAtlasData.js HEADs the JPEGs and
// persists static/data_browser/renderSizes.json, which the runtime path
// imports) — if ANY variant is missing a size or um, the finest-first order is
// kept unchanged, so brand-new variants degrade to the old behavior until the
// next build refreshes the sizes file.
function orderByContent(variants, renderSizes) {
  if (!Array.isArray(variants) || variants.length < 2) return variants;
  const sizes = renderSizes || {};
  const scores = variants.map((v) => {
    const bytes = v.thumbUrl != null ? sizes[v.thumbUrl] : null;
    return typeof bytes === "number" && bytes > 0 && v.um != null
      ? bytes * v.um * v.um
      : null;
  });
  if (scores.some((s) => s == null)) return variants;
  return variants
    .map((v, i) => [v, scores[i]])
    .sort((a, b) => b[1] - a[1]) // most content first; stable, ties keep um order
    .map(([v]) => v);
}

// Generic per-segment render-variant extractor shared by the 2D ink-detection
// and 3D alpha-render sides: filter `data` by `type`, resolve + collapse to
// one surviving entry per target volume, then join each survivor's matching
// downsampled counterpart (`downType` — see downFor) as a thumbUrl.
// `one` is the PRIMARY variant — the most-content one when renderSizes covers
// every variant (see orderByContent), the finest otherwise; exact legacy
// full/alpha semantics are unchanged for the (overwhelmingly common)
// single-variant case. `variants` is the complete list, in the same order,
// and is only set when there's more than one to compare.
function extractRenderVariants(data, type, downType, s3ToHttp, volumes, renderSizes) {
  const fulls = data.filter((d) => d.type === type).map((d) => toRenderMeta(d, s3ToHttp, volumes));
  const downs = data
    .filter((d) => d.type === downType)
    .map((d) => toRenderMeta(d, s3ToHttp, volumes));
  const grouped = groupRendersByVolume(fulls);
  const variants = orderByContent(
    grouped.map((g) => {
      const down = downFor(g, downs, grouped.length);
      return { ...g, thumbUrl: down ? down.url : null };
    }),
    renderSizes
  );
  const result = { one: variants[0] || null };
  if (variants.length > 1) result.variants = variants;
  return result;
}

// Per-segment ink-detection predictions for a sample, as public HTTPS links —
// the per-segment list the old atlas exposed. One entry per segment (its
// primary run — see orderByContent) with a downsampled preview for
// thumbnailing. Embargoed scrolls expose nothing. `s3ToHttp` maps a segment's
// s3:// access-root to a public https URL via the dataAccess rewrites.
// Segments with more than one multi-volume render variant additionally get
// `renders`/`alphaRenders` arrays (see extractRenderVariants) feeding the
// lightbox compare view; the pre-existing fields keep pointing at the primary
// (first) variant either way.
function extractInkSegments(sample, id, s3ToHttp, renderSizes) {
  if (EMBARGOED.has(id)) return [];
  const segs = (sample && sample.segments) || {};
  const volumes = (sample && sample.volumes) || {};
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
    // 2D ink-detection: the flattened prediction image(s) + downsampled
    // preview(s), one variant per target volume.
    const inkX = extractRenderVariants(
      data,
      "ink-detection",
      "ink-detection-downsampled",
      s3ToHttp,
      volumes,
      renderSizes
    );
    // 3D ink (alpha-render): ink painted on the rendered surface, same shape.
    // The full .tif is tens of MB (download only); the ds8 .jpg is the
    // Thumbor thumbnail src. s3ToHttp so the jpg matches the thumbnail prefix
    // and the tif is clickable.
    const alphaX = extractRenderVariants(
      data,
      "alpha-render",
      "alpha-render-downsampled",
      s3ToHttp,
      volumes,
      renderSizes
    );
    const full = inkX.one;
    const alphaOne = alphaX.one;
    // Emit a segment if it has EITHER a 2D ink prediction OR a 3D-ink preview
    // (so the 3 alpha-only PHercParis4 segments still appear).
    if (!full && !(alphaOne && alphaOne.thumbUrl)) continue;
    const layers = oneUrl(data.find((d) => d.type === "layers-zarr"));
    // Segment root folder (holds tifxyz/, surface-volumes/, ink-detection/…),
    // derived from any of the segment's URLs and kept as an s3:// URI. The
    // gallery renders it as a file-browser link + a copy-for-rclone button —
    // community feedback: the old per-segment .obj link was useless (no tool
    // consumes .obj) and the tifxyz was hard to find from the browser.
    const segFolder = (u) => {
      const m = String(u || "").match(/^(.*?\/segments\/[^/]+\/)/);
      return m ? m[1] : null;
    };
    const folder =
      segFolder(oneUrl(data.find((d) => d.type === "obj-flattened" || d.type === "obj"))) ||
      segFolder(layers) ||
      segFolder(oneUrl(data.find((d) => d.type === "ink-detection"))) ||
      segFolder(oneUrl(data.find((d) => d.type === "alpha-render")));
    out.push({
      id: key,
      label: inkSegmentLabel(sid),
      um: full ? full.um : null,
      full: full ? full.url : null,
      down: full ? full.thumbUrl : null,
      preview: full ? full.thumbUrl || full.url : null,
      alpha: alphaOne ? alphaOne.url : null,
      alphaPrev: alphaOne ? alphaOne.thumbUrl : null,
      layers,
      folder,
      ...(inkX.variants ? { renders: inkX.variants } : {}),
      ...(alphaX.variants ? { alphaRenders: alphaX.variants } : {}),
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

// Human-readable inventory label: "PHerc0172" -> "PHerc. 172",
// "PHerc0175A" -> "PHerc. 175A", "PHercParis4" -> "PHerc. Paris 4".
// Non-PHerc ids pass through unchanged; routes/URLs keep the raw id.
function phercLabel(id) {
  const m = /^PHerc(.+)$/.exec(id || "");
  if (!m) return id;
  const rest = m[1];
  const letters = /^([A-Za-z]+)(.*)$/.exec(rest);
  if (letters) {
    const tail = letters[2] ? ` ${letters[2].replace(/^0+(?=\d)/, "")}` : "";
    return `PHerc. ${letters[1]}${tail}`;
  }
  return `PHerc. ${rest.replace(/^0+(?=\d)/, "")}`;
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
    label: phercLabel(id),
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
 *   - renderSizes:  parsed static/data_browser/renderSizes.json ({downsampled
 *                   render url: bytes}) — picks each multi-variant ink
 *                   segment's primary render (see orderByContent); optional,
 *                   omitting it keeps the finest-um-first order everywhere
 */
function buildIndex(samples, opts) {
  const { overlay = {}, meshManifest = {}, rewrites = [], renderSizes = {} } = opts || {};
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
      ? extractInkSegments(samples[id], id, s3ToHttp, renderSizes)
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
