// Shared helpers for the /get_started interactive demos.
// Everything here is browser-only; call from effects/handlers, never at
// module scope (Docusaurus SSR).

import { useEffect, useState } from "react";

export function loadImage(src) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error(`failed to load ${src}`));
    img.src = src;
  });
}

export async function loadManifest(baseUrl) {
  const res = await fetch(`${baseUrl}manifest.json`);
  if (!res.ok) throw new Error(`manifest ${res.status}`);
  return res.json();
}

// Draw an image and return its single-channel pixel data (R channel).
export function imageToGray(img, w, h) {
  const c = document.createElement("canvas");
  c.width = w;
  c.height = h;
  const ctx = c.getContext("2d", { willReadFrequently: true });
  ctx.drawImage(img, 0, 0, w, h);
  const { data } = ctx.getImageData(0, 0, w, h);
  const gray = new Uint8ClampedArray(w * h);
  for (let i = 0; i < gray.length; i++) gray[i] = data[i * 4];
  return gray;
}

export function makeCanvas(w, h) {
  const c = document.createElement("canvas");
  c.width = w;
  c.height = h;
  return c;
}

// Static film-grain noise tile used for "model uncertainty".
export function makeNoiseCanvas(w, h) {
  const c = makeCanvas(w, h);
  const ctx = c.getContext("2d");
  const img = ctx.createImageData(w, h);
  for (let i = 0; i < w * h; i++) {
    const v = 30 + Math.random() * 60;
    img.data[i * 4] = v;
    img.data[i * 4 + 1] = v;
    img.data[i * 4 + 2] = v;
    img.data[i * 4 + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);
  return c;
}

// Pointer position in canvas IMAGE coordinates (canvas is CSS-scaled).
export function eventToImageXY(e, canvas) {
  const rect = canvas.getBoundingClientRect();
  const x = ((e.clientX - rect.left) / rect.width) * canvas.width;
  const y = ((e.clientY - rect.top) / rect.height) * canvas.height;
  return [x, y];
}

// what the affordance pills should say on this device
export function pillLabel() {
  return typeof window !== "undefined" &&
    window.matchMedia?.("(pointer: coarse)").matches
    ? "Tap here"
    : "Click here";
}

// true on touch-first devices. Resolved in an effect so SSR and the first
// client render agree (both say false), then touch devices flip after mount.
export function useCoarsePointer() {
  const [coarse, setCoarse] = useState(false);
  useEffect(() => {
    setCoarse(window.matchMedia?.("(pointer: coarse)").matches ?? false);
  }, []);
  return coarse;
}

// "click/tap here" affordance drawn in canvas space next to a target:
// dark pill, orange border and text (the theme colors), auto-flipping to
// the left when it would leave the image. `u` = image-space units per CSS
// px, so the pill keeps a constant on-screen size at any zoom.
export function drawPill(ctx, label, x, y, offset, u, maxX) {
  const fs = 13 * u;
  ctx.font = `600 ${fs}px system-ui, -apple-system, sans-serif`;
  const tw = ctx.measureText(label).width;
  const padX = 8 * u;
  const ph = fs * 1.9;
  const gap = offset + 12 * u;
  let bx = x + gap;
  if (bx + tw + padX * 2 > maxX) bx = x - gap - tw - padX * 2;
  ctx.beginPath();
  if (ctx.roundRect) ctx.roundRect(bx, y - ph / 2, tw + padX * 2, ph, ph / 2);
  else ctx.rect(bx, y - ph / 2, tw + padX * 2, ph);
  ctx.fillStyle = "rgba(18,19,22,0.92)";
  ctx.fill();
  ctx.strokeStyle = "rgba(229,80,43,0.9)";
  ctx.lineWidth = 1.2 * u;
  ctx.stroke();
  ctx.fillStyle = "#ff6a3d";
  const tb = ctx.textBaseline;
  ctx.textBaseline = "middle";
  ctx.fillText(label, bx + padX, y + fs * 0.05);
  ctx.textBaseline = tb;
}
