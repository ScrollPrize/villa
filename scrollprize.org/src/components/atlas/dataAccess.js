// dataAccess — resolve Vesuvius data URLs the way the old prebuilt atlas did.
//
// Ported from the old atlas's vendored bundle (static/atlas on main, removed by
// this redesign). The config lives in src/data/atlasDataAccess.json. Three jobs:
//   1. rewrite s3:// URIs to public HTTPS (esp. s3://vesuvius-challenge/ ->
//      https://data.aws.ash2txt.org/samples/, the full-dataset gateway);
//   2. build a Neuroglancer deep-link for an OME-Zarr volume via the zarr proxy
//      (the old atlas's `Ef` + `Jf`/`Ku` — it never rendered zarr in-page, it
//      handed a zarr2:// source to Neuroglancer);
//   3. build a Thumbor thumbnail URL for a bucket image (the old atlas's `yv`).
//
// NOTE: Neuroglancer reads these OME-Zarr stores directly from the bucket (both
// data buckets send Access-Control-Allow-Origin: *) and decodes Blosc — LZ4 and
// Zstd — natively, so we no longer route through the zarr.aws.ash2txt.org proxy
// (which only handled LZ4 and stalled the Blosc-Zstd surface / CT / 3D-ink
// prediction overlays).

import cfg from "@site/src/data/atlasDataAccess.json";

const DA = cfg.dataAccess || {};
const NEUROGLANCER = "https://neuroglancer-demo.appspot.com/#!";

// Apply the configured s3://->https rewrites. Returns the url unchanged if no
// rule matches (or it is already https).
export function toHttp(url) {
  if (!url) return url;
  for (const r of DA.rewrites || []) {
    if (url.startsWith(r.from)) return r.to + url.slice(r.from.length);
  }
  return url;
}

// Reverse of toHttp: present a copy-pasteable s3:// URI for an https URL.
export function toS3(url) {
  if (!url) return url;
  for (const r of DA.rewrites || []) {
    if (url.startsWith(r.to)) return r.from + url.slice(r.to.length);
  }
  return url;
}

// Build a browsable S3 index URL for a directory-like key (e.g. a `.zarr/`).
// A direct GET on a prefix returns S3 "NoSuchKey", so for zarr stores etc. we
// link to the bucket's index.html browser instead. Uses the configured
// browser.indexTemplate; falls back to plain HTTPS when no browser is set or
// the URL isn't under the browsable bucket.
export function browseUrl(url) {
  const https = toHttp(url);
  const browser = DA.browser || {};
  if (!browser.base || !browser.indexTemplate || !https.startsWith(browser.base)) {
    return https;
  }
  const path = https.slice(browser.base.length);
  return browser.indexTemplate.replace("{path}", path);
}

// Build the Neuroglancer zarr source for a volume. Both data buckets serve the
// OME-Zarr with open CORS, so we hand Neuroglancer the bucket URL directly
// (`zarr2://https://…`) and let it read the multiscale group and decode Blosc
// (LZ4 or Zstd) itself — bypassing the LZ4-only zarr proxy. The trailing slash
// is trimmed so Neuroglancer builds `…zarr/.zgroup`, not `…zarr//.zgroup`.
// (The `dataAccess.zarrProxy` config is retained for reference but unused.)
function zarrSource(volumeUrl) {
  const https = toHttp(volumeUrl);
  if (!https) return null;
  return `zarr2://${https.replace(/\/+$/, "")}`;
}

// Build a Neuroglancer deep-link that opens a single grayscale image layer for
// the given OME-Zarr volume (s3:// or https). Returns null if no proxy rule
// applies. Mirrors the old atlas's single-layer state (`Jf`), 4-panel layout.
export function neuroglancerUrl(volumeUrl, name = "volume") {
  const source = zarrSource(volumeUrl);
  if (!source) return null;
  const state = {
    layers: [
      {
        type: "image",
        source,
        name,
        shader:
          "#uicontrol invlerp normalized\nvoid main() { emitGrayscale(normalized()); }",
      },
    ],
    layout: "4panel",
  };
  return NEUROGLANCER + encodeURIComponent(JSON.stringify(state));
}

// Build a Neuroglancer deep-link that overlays a prediction on the raw CT volume
// it was computed from: the CT as a grayscale image layer, the prediction as a
// colored additive layer on top. Returns null if the prediction has no usable
// source; falls back to a prediction-only view when no base CT is given.
// Mirrors the old atlas's multi-layer prediction "Overlay".
export function neuroglancerOverlayUrl(baseUrl, predUrl, names = {}) {
  const predSource = zarrSource(predUrl);
  if (!predSource) return null;
  const baseSource = zarrSource(baseUrl);
  const layers = [];
  if (baseSource) {
    layers.push({
      type: "image",
      source: baseSource,
      name: names.base || "CT volume",
      shader:
        "#uicontrol invlerp normalized\nvoid main() { emitGrayscale(normalized()); }",
    });
  }
  layers.push({
    type: "image",
    source: predSource,
    name: names.pred || "prediction",
    blend: "additive",
    shader:
      '#uicontrol vec3 color color(default="#ff7a1a")\n#uicontrol invlerp normalized\nvoid main() { emitRGBA(vec4(color, normalized())); }',
  });
  return NEUROGLANCER + encodeURIComponent(JSON.stringify({ layers, layout: "4panel" }));
}

// Build Thumbor thumbnail URLs for a bucket image (old atlas `yv`). Returns
// { cacheUrl, serviceUrl }: try the pre-baked S3 cache first, fall back to the
// live Thumbor service on <img onError>. Both null if no thumbnail server
// matches the image's prefix.
// Optimized hero URL for an item's physical photo, via the Thumbor service
// (host normalized so it matches a thumbnails[] prefix). Falls back to the
// original URL if no thumbnail server matches.
export function photoThumb(photoUrl, width = 1400) {
  if (!photoUrl) return null;
  const norm = photoUrl.replace(
    "vesuvius-challenge-open-data.s3.amazonaws.com",
    "vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com"
  );
  const { serviceUrl } = thumbnailUrls(norm, width, width);
  return serviceUrl || photoUrl;
}

export function thumbnailUrls(imageUrl, width = 400, height = 400, fmt = "webp") {
  const t = (DA.thumbnails || []).find((s) => imageUrl && imageUrl.startsWith(s.prefix));
  if (!t) return { cacheUrl: null, serviceUrl: null };
  const path = imageUrl.slice(t.prefix.length);
  const serviceUrl = `${t.serviceBase}unsafe/fit-in/${width}x${height}/filters:format(${fmt})/${path}`;
  // The S3 cache key double-encodes the colon in `filters:` (old atlas behavior).
  const cacheUrl = t.cacheBase
    ? `${t.cacheBase}fit-in/${width}x${height}/filters%253Aformat(${fmt})/${path}`
    : serviceUrl;
  return { cacheUrl, serviceUrl };
}
