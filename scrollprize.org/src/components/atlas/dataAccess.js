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
// NOTE: the zarr proxy (zarr.aws.ash2txt.org) currently only decodes Blosc-LZ4;
// the surface-prediction volumes are Blosc-Zstd, so their Neuroglancer links
// won't load until the proxy gains Zstd support. LZ4 volumes work today.

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

// Build the zarr proxy source string for a volume (old atlas `Ef`). Picks the
// first zarrProxy rule whose `match` substring appears in the https URL and
// rewrites it to `zarr2://{proxyBase}/{pathAfterMatch}`.
function zarrSource(volumeUrl) {
  const https = toHttp(volumeUrl);
  const { rules = [], defaultProxyBase } = DA.zarrProxy || {};
  for (const { match, proxyBase } of rules) {
    if (https.includes(match)) {
      const after = https.split(match)[1];
      return `zarr2://${proxyBase}/${after}`;
    }
  }
  return defaultProxyBase ? `zarr2://${defaultProxyBase}/${https}` : null;
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
