/**
 * rehype plugin: stamp intrinsic width/height (+ lazy/async decoding) onto
 * content images so the browser reserves layout space and avoids Cumulative
 * Layout Shift (CLS). Standalone images also get a `zoomable` class that the
 * imageZoom client module uses for a click-to-full-resolution lightbox.
 *
 * Images nested inside an <a> are deliberately NOT marked `zoomable`: their
 * click should follow the authored link, not open the lightbox (which would
 * otherwise double-fire — zoom + navigate). They still receive width/height.
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

function isAnchor(node) {
  return (
    (node.type === "element" && node.tagName === "a") ||
    ((node.type === "mdxJsxFlowElement" || node.type === "mdxJsxTextElement") &&
      node.name === "a")
  );
}

// Depth-first walk tracking whether we're inside an <a> ancestor.
function walk(node, inLink, fn) {
  if (!node || typeof node !== "object") return;
  fn(node, inLink);
  const nextInLink = inLink || isAnchor(node);
  const kids = node.children;
  if (Array.isArray(kids)) for (const c of kids) walk(c, nextInLink, fn);
}

function handleHastImg(node, inLink) {
  const props = node.properties || (node.properties = {});
  const dims = getDims(props.src);
  if (dims) {
    if (props.width == null) props.width = dims[0];
    if (props.height == null) props.height = dims[1];
  }
  if (props.loading == null) props.loading = "lazy";
  if (props.decoding == null) props.decoding = "async";
  if (!inLink) {
    const cls = Array.isArray(props.className)
      ? props.className
      : props.className
        ? [props.className]
        : [];
    if (!cls.includes("zoomable")) cls.push("zoomable");
    props.className = cls;
  }
}

function handleJsxImg(node, inLink) {
  const attrs = node.attributes || (node.attributes = []);
  const attr = (n) =>
    attrs.find((a) => a.type === "mdxJsxAttribute" && a.name === n);
  const has = (n) => Boolean(attr(n));
  const push = (name, value) =>
    attrs.push({ type: "mdxJsxAttribute", name, value });

  const srcAttr = attr("src");
  const src =
    srcAttr && typeof srcAttr.value === "string" ? srcAttr.value : undefined;
  const dims = getDims(src);
  if (dims) {
    if (!has("width")) push("width", String(dims[0]));
    if (!has("height")) push("height", String(dims[1]));
  }
  if (!has("loading")) push("loading", "lazy");
  if (!has("decoding")) push("decoding", "async");

  if (!inLink) {
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
}

module.exports = function rehypeImageDimensions() {
  return (tree) => {
    walk(tree, false, (node, inLink) => {
      if (node.type === "element" && node.tagName === "img") {
        handleHastImg(node, inLink);
      } else if (
        (node.type === "mdxJsxFlowElement" ||
          node.type === "mdxJsxTextElement") &&
        node.name === "img"
      ) {
        handleJsxImg(node, inLink);
      }
    });
  };
};
