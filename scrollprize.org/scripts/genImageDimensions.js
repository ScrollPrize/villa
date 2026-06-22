#!/usr/bin/env node
/**
 * Build-time image dimension manifest generator.
 *
 * Walks static/img and records intrinsic width/height for every raster image,
 * keyed by its web path (e.g. "/img/data/rep_norms_10037.png"). The rehype
 * image-dimensions plugin reads this manifest to stamp width/height on <img>
 * tags so the browser can reserve space and avoid layout shift (CLS).
 *
 * Uses ImageMagick `identify` (already available system-wide); no npm deps.
 * Output: .image-dimensions.json at the project root (gitignored, regenerated
 * on every build via the prebuild/prestart hooks).
 */
const { execFileSync } = require("child_process");
const fs = require("fs");
const path = require("path");

const ROOT = path.resolve(__dirname, "..");
const STATIC_DIR = path.join(ROOT, "static");
const IMG_DIR = path.join(STATIC_DIR, "img");
const OUT = path.join(ROOT, ".image-dimensions.json");

function main() {
  if (!fs.existsSync(IMG_DIR)) {
    fs.writeFileSync(OUT, "{}");
    return;
  }
  // find + xargs + identify; one line per (frame of an) image: "path\tW\tH"
  const cmd =
    `find "${IMG_DIR}" -type f ` +
    `\\( -iname '*.png' -o -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.webp' -o -iname '*.gif' \\) ` +
    `-print0 | xargs -0 -n 40 identify -format '%i|%w|%h\\n' 2>/dev/null`;

  let out = "";
  try {
    out = execFileSync("bash", ["-c", cmd], {
      maxBuffer: 1024 * 1024 * 128,
    }).toString();
  } catch (e) {
    // identify can exit non-zero on a bad file while still emitting good lines
    out = (e.stdout || "").toString();
  }

  const dims = {};
  for (const line of out.split("\n")) {
    if (!line) continue;
    const parts = line.split("|");
    if (parts.length < 3) continue;
    const [file, w, h] = parts;
    const clean = file.replace(/\[\d+\]$/, ""); // strip animated-frame suffix
    let rel = clean.startsWith(STATIC_DIR) ? clean.slice(STATIC_DIR.length) : clean;
    rel = rel.split(path.sep).join("/");
    if (!rel.startsWith("/")) rel = "/" + rel;
    if (dims[rel]) continue; // keep first frame for animated images
    const wi = parseInt(w, 10);
    const hi = parseInt(h, 10);
    if (wi > 0 && hi > 0) dims[rel] = [wi, hi];
  }

  fs.writeFileSync(OUT, JSON.stringify(dims));
  console.log(
    `[image-dimensions] wrote ${Object.keys(dims).length} entries to ${path.relative(ROOT, OUT)}`,
  );
}

main();
