// What the pages share: fetching, the patch pictures, and the cross-section that shows
// where a patch sits. From the first runs' review_common.js; the pictures are now the ones
// the stages stored, under /img/<pair>/.

const MODES = ["overlay", "flip", "checker", "coarse", "fine"];
const LETTERS = "ABCDEFGH";
const $ = id => document.getElementById(id);
const fmt = (v, d = 2) => v == null ? "–" : v.toFixed(d);
const escapeHtml = text => String(text).replace(/[&<>"]/g, c => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;"})[c]);
// The pair a page is about: /pairs/<pair>/<page>.
const PAIR = decodeURIComponent(location.pathname.split("/")[2] || "");

async function call(path, body) {
  const response = await fetch(path, body === undefined ? {} :
    {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(body)});
  const value = await response.json();
  if (!response.ok) throw new Error(value.error || response.statusText);
  return value;
}

function picture(url) {
  return new Promise((resolve, reject) => { const im = new Image();
    im.onload = () => { const c = document.createElement("canvas"); c.width = im.width; c.height = im.height;
      const g = c.getContext("2d"); g.drawImage(im, 0, 0); resolve(g.getImageData(0, 0, im.width, im.height)); };
    im.onerror = reject; im.src = url; });
}

// The coarse and fine strips as `show` asks -- "overlay" (red coarse, cyan fine, grey where
// they agree), "checker", or one side alone -- enlarged `zoom` times.
function drawStrip(canvas, coarse, fine, show, zoom = 3) {
  const w = coarse.width, h = coarse.height;
  const out = new ImageData(w, h), a = coarse.data, b = fine.data, o = out.data;
  for (let i = 0; i < w * h; i++) {
    const k = i * 4, ca = a[k], fb = b[k];
    let r, g, bl;
    if (show === "overlay") { r = ca; g = fb; bl = fb; }
    else if (show === "checker") { const x = i % w, y = (i / w) | 0;
      const v = (((x >> 4) + (y >> 4)) & 1) ? fb : ca; r = g = bl = v; }
    else { const v = show === "coarse" ? ca : fb; r = g = bl = v; }
    o[k] = r; o[k + 1] = g; o[k + 2] = bl; o[k + 3] = 255;
  }
  const small = document.createElement("canvas"); small.width = w; small.height = h;
  small.getContext("2d").putImageData(out, 0, 0);
  canvas.width = w * zoom; canvas.height = h * zoom;
  const g = canvas.getContext("2d"); g.imageSmoothingEnabled = false;
  g.drawImage(small, 0, 0, w * zoom, h * zoom);
}

// The stored cross-section through a patch's middle, with the unit boxed on it.
async function drawSection(img, box, patch, origin, size) {
  box.hidden = true;
  let info;
  try { info = await (await fetch(`/img/${PAIR}/holdout/${patch}/section.json`)).json(); }
  catch (e) { img.hidden = true; return; }
  img.hidden = false;
  img.src = `/img/${PAIR}/holdout/${patch}/section.png`;
  img.onload = () => {
    const k = img.clientWidth / img.naturalWidth, low = info.low_yx || [0, 0];
    const y = (origin[1] - low[0] + 0.5) / info.scale - 0.5, x = (origin[2] - low[1] + 0.5) / info.scale - 0.5;
    const side = Math.max(size / info.scale * k, 4);
    Object.assign(box.style, {left: `${x * k}px`, top: `${y * k}px`, width: `${side}px`, height: `${side}px`});
    box.hidden = false;
  };
}

// The modes row under a strip; `render` redraws after a change.
function modeButtons(holder, mode, flipped, show, onPick) {
  holder.innerHTML = MODES.map(m => `<button class="${m === mode ? "on" : ""}" data-m="${m}">${m}${
    m === "flip" && mode === "flip" ? (flipped ? ": fine" : ": coarse") : ""}</button>`).join("") +
    `<span class="muted">${show === "overlay" ? "red = coarse volume, cyan = fine volume; grey where they agree" : ""}</span>`;
  for (const b of holder.querySelectorAll("button")) b.onclick = () => onPick(b.dataset.m);
}
