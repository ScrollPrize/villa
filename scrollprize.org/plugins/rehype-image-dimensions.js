/**
 * rehype plugin: stamp intrinsic width/height (+ lazy/async decoding + a
 * `zoomable` class) onto content images so the browser reserves layout space
 * and avoids Cumulative Layout Shift (CLS). Dimensions come from the build-time
 * manifest produced by scripts/genImageDimensions.js.
 *
 * Handles BOTH markdown images (`![](...)` -> hast `element` <img>) and raw
 * JSX `<img>` written in .md/.mdx (-> `mdxJsxFlowElement`/`mdxJsxTextElement`).
 *
 * Plain CommonJS with a manual tree walk (no unist-util-visit) to avoid any
 * ESM/CJS interop issues when required from the CommonJS Docusaurus config.
 */
const fs = require("fs");
const path = require("path");

const DIMS_PATH = path.resolve(__dirname, "..", ".image-dimensions.json");
let DIMS = {};
try {
  DIMS = JSON.parse(fs.readFileSync(DIMS_PATH, "utf8"));
} catch (e) {
  DIMS = {}; // manifest absent -> plugin is a no-op for dimensions
}

function normalizeSrc(src) {
  if (!src || typeof src !== "string") return null;
  if (/^(https?:)?\/\//.test(src) || src.startsWith("data:")) return null;
  const s = src.split("#")[0].split("?")[0];
  if (!s.startsWith("/")) return null; // only absolute /img/... paths
  return s;
}

function getDims(src) {
  const s = normalizeSrc(src);
  if (!s) return null;
  return DIMS[s] || null;
}

function walk(node, fn) {
  if (!node || typeof node !== "object") return;
  fn(node);
  const kids = node.children;
  if (Array.isArray(kids)) for (const c of kids) walk(c, fn);
}

function handleHastImg(node) {
  const props = node.properties || (node.properties = {});
  const dims = getDims(props.src);
  if (dims) {
    if (props.width == null) props.width = dims[0];
    if (props.height == null) props.height = dims[1];
  }
  if (props.loading == null) props.loading = "lazy";
  if (props.decoding == null) props.decoding = "async";
  const cls = Array.isArray(props.className)
    ? props.className
    : props.className
      ? [props.className]
      : [];
  if (!cls.includes("zoomable")) cls.push("zoomable");
  props.className = cls;
}

function handleJsxImg(node) {
  const attrs = node.attributes || (node.attributes = []);
  const attr = (n) =>
    attrs.find((a) => a.type === "mdxJsxAttribute" && a.name === n);
  const has = (n) => Boolean(attr(n));
  const push = (name, value) =>
    attrs.push({ type: "mdxJsxAttribute", name, value });

  const srcAttr = attr("src");
  const src = srcAttr && typeof srcAttr.value === "string" ? srcAttr.value : undefined;
  const dims = getDims(src);
  if (dims) {
    if (!has("width")) push("width", String(dims[0]));
    if (!has("height")) push("height", String(dims[1]));
  }
  if (!has("loading")) push("loading", "lazy");
  if (!has("decoding")) push("decoding", "async");

  const clsAttr = attrs.find(
    (a) =>
      a.type === "mdxJsxAttribute" &&
      (a.name === "className" || a.name === "class"),
  );
  if (clsAttr) {
    if (
      typeof clsAttr.value === "string" &&
      !clsAttr.value.split(/\s+/).includes("zoomable")
    ) {
      clsAttr.value = `${clsAttr.value} zoomable`;
    }
  } else {
    push("className", "zoomable");
  }
}

module.exports = function rehypeImageDimensions() {
  return (tree) => {
    walk(tree, (node) => {
      if (node.type === "element" && node.tagName === "img") {
        handleHastImg(node);
      } else if (
        (node.type === "mdxJsxFlowElement" ||
          node.type === "mdxJsxTextElement") &&
        node.name === "img"
      ) {
        handleJsxImg(node);
      }
    });
  };
};
