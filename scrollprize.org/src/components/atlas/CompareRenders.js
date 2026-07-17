import React, { useRef, useState } from "react";
import { thumbnailUrls } from "./dataAccess";

// CompareRenders — an in-lightbox swipe/sweep comparison of exactly two render
// variants of the same ink segment (either two `renders` 2D-ink entries or two
// `alphaRenders` 3D-ink entries — see buildIndex.js extractRenderVariants for
// the shared { url, thumbUrl, um, energy, targetVolume, modelId, modelName,
// renderLevel } shape). Drag (or arrow-key-nudge) a vertical divider to reveal
// more of one side or the other, like a classic before/after slider.
//
// Pointer Events (not mouse+touch listeners) drive the drag: onPointerDown
// captures the pointer on the frame itself and jumps the divider to the click
// position; onPointerMove just checks `e.buttons !== 0` — no isDragging state,
// no document-level listeners to leak. Both <img> layers are pointer-events:
// none (custom.css) so the frame div is the sole hit/drag target — this also
// stops the browser's native image-drag ghost. touch-action: none on the frame
// stops touch scroll from hijacking the drag.

// "1.13" -> "1.13", "2.00" -> "2", "2.50" -> "2.5" — toFixed() padding trimmed
// back off so whole/short numbers don't grow a fake-precision tail.
function trimNum(s) {
  if (!s.includes(".")) return s;
  return s.replace(/0+$/, "").replace(/\.$/, "");
}

// "1.13µm · 59keV" — only the parts that exist, joined; "variant" if neither.
function formatVariantLabel(v) {
  const parts = [];
  if (v && v.um != null) parts.push(`${trimNum(v.um.toFixed(2))}µm`);
  if (v && v.energy != null) parts.push(`${trimNum(String(v.energy))}keV`);
  return parts.join(" · ") || "variant";
}

const tail = (s, n = 6) => (s ? String(s).slice(-n) : "");

// Real edge case (one PHerc0172 segment has two near-duplicate volumes sharing
// the same um/energy): if both sides format identically, disambiguate with a
// short tail of each side's targetVolume so the two labels read distinctly.
// targetVolume is the primary disambiguator, but it (and even modelId) can be
// absent depending on which metadata source populated this render — a segment
// on the live (non-bundled) index has been observed with both null — so fall
// through modelId, then the render's own url, which is always distinct between
// two different variants; this guarantees the two labels are never identical.
function variantLabels(renders) {
  const labels = renders.map(formatVariantLabel);
  if (labels[0] !== labels[1]) return labels;
  const keysFor = (v) => [tail(v?.targetVolume), tail(v?.modelId), tail(v?.url)];
  const keys = renders.map(keysFor);
  for (let col = 0; col < 3; col++) {
    const [ka, kb] = [keys[0][col], keys[1][col]];
    if (ka && kb && ka !== kb) {
      return labels.map((l, i) => `${l} (…${keys[i][col]})`);
    }
  }
  // Should be unreachable (would mean genuinely identical variants) — never
  // hand back two indistinguishable labels regardless.
  return labels.map((l, i) => `${l} #${i + 1}`);
}

const clamp = (n, lo, hi) => Math.min(hi, Math.max(lo, n));

// Exactly two variants are compared at a time. `renders`/`alphaRenders` are
// pre-sorted primary-first by buildIndex.js, so when more than two are ever
// supplied (never observed in real data as of writing — confirmed max is 2
// target_volume variants per segment), fall back to the first two rather
// than rendering nothing — see the InkSegmentsGallery toggle label, which
// says "(2 of N)" in that case so it's clear only a pair is being compared.
function comparePair(renders) {
  if (!Array.isArray(renders) || renders.length < 2) return null;
  return renders.length === 2 ? renders : renders.slice(0, 2);
}

// The <img> src for one side: the variant's downsampled render, resized
// through the same Thumbor proxy every other gallery image uses (the single
// lightbox view requests the identical 2000x2000 fit-in). The raw ds8 JPEGs
// are multi-MB S3 objects served as binary/octet-stream — measured 0.7–1.3MB
// per side on PHerc0139 vs ~60KB webp via the proxy — slow enough that the
// second layer looked "never loaded" during a sweep. Falls back to the raw
// URL when no proxy prefix matches (thumbnailUrls returns null serviceUrl).
const sideSrc = (v) =>
  (v && v.thumbUrl && thumbnailUrls(v.thumbUrl, 2000, 2000).serviceUrl) ||
  (v ? v.thumbUrl : null);

export default function CompareRenders({ renders, label }) {
  const frameRef = useRef(null);
  const [pos, setPos] = useState(50); // divider position, percent from the left

  const pair = comparePair(renders);
  if (!pair) return null;
  const [a, b] = pair;
  const [labelA, labelB] = variantLabels(pair);

  const setFromClientX = (clientX) => {
    const el = frameRef.current;
    if (!el) return;
    const rect = el.getBoundingClientRect();
    if (!rect.width) return;
    setPos(clamp(((clientX - rect.left) / rect.width) * 100, 0, 100));
  };

  const onPointerDown = (e) => {
    e.currentTarget.setPointerCapture(e.pointerId);
    setFromClientX(e.clientX);
  };
  const onPointerMove = (e) => {
    if (e.buttons !== 0) setFromClientX(e.clientX);
  };
  const onKeyDown = (e) => {
    if (e.key === "ArrowLeft") {
      e.preventDefault();
      e.stopPropagation();
      setPos((p) => clamp(p - 2, 0, 100));
    } else if (e.key === "ArrowRight") {
      e.preventDefault();
      e.stopPropagation();
      setPos((p) => clamp(p + 2, 0, 100));
    }
  };

  // Top layer (labelA / left side) revealed left of the divider, clipped away
  // to the right of it.
  const clipTop = `inset(0 ${100 - pos}% 0 0)`;

  return (
    <div className="comparewrap">
      <div
        ref={frameRef}
        className="compareframe"
        role="slider"
        tabIndex={0}
        aria-label={`Compare renders — ${label || "segment"}`}
        aria-valuenow={Math.round(pos)}
        aria-valuemin={0}
        aria-valuemax={100}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onKeyDown={onKeyDown}
      >
        <img
          className="cmpimg-b"
          src={sideSrc(b)}
          alt={`${label || ""} — ${labelB}`}
          draggable={false}
        />
        <img
          className="cmpimg-a"
          src={sideSrc(a)}
          alt={`${label || ""} — ${labelA}`}
          draggable={false}
          style={{ clipPath: clipTop }}
        />
        <div className="cmpdivider" style={{ left: `${pos}%` }}>
          <div className="cmphandle" data-testid="sweep-handle" />
        </div>
        <span className="cmplabel-left">{labelA}</span>
        <span className="cmplabel-right">{labelB}</span>
      </div>
      <div className="cmpfoot">
        {a.url ? (
          <a href={a.url} target="_blank" rel="noopener noreferrer">
            {labelA} · full TIFF ↗
          </a>
        ) : null}
        {b.url ? (
          <a href={b.url} target="_blank" rel="noopener noreferrer">
            {labelB} · full TIFF ↗
          </a>
        ) : null}
      </div>
    </div>
  );
}
