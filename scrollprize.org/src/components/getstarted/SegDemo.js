import React, { useEffect, useRef, useState } from "react";
import useBaseUrl from "@docusaurus/useBaseUrl";
import {
  loadImage,
  loadManifest,
  imageToGray,
  eventToImageXY,
  drawPill,
  pillLabel,
  useCoarsePointer,
} from "./demoUtils";
import UnrollReveal from "./UnrollReveal";

/*
 * "Trace the sheet" — live segmentation on the real Scroll 1 sheet the
 * ink demo's word was read from. The flow is guided end to end by pulsing
 * markers (all of them honor the Hints slider; at 0 they disappear):
 *
 * Left: the real scroll as a textured 3D turntable (drag to rotate; tap a
 *       plane ring to open it) — a z band of the parent segment, stretched
 *       tall (manifest zStretch), cut away helically so the windings show.
 * Middle: the FULL CT cross-section for the active plane, opened zoomed
 *         on the start of the target sheet (manifest introZoom) with the
 *         first dot already placed and a pulsing marker one step ahead;
 *         a hi-res detail inset covers the trace region; the view glides
 *         along as the trace grows. Pinch/scroll zooms, drag pans.
 * Right: the flattened surface render (the real ΟΡΦ crop, native aspect).
 *        Each dot's distance to the true surface picks which real
 *        normal-offset render shows in that region: on-surface -> crisp;
 *        off-surface -> off-center papyrus, then the gap, then the
 *        neighboring layer; no dots -> dark.
 *
 * Data contract: static/get-started/seg/manifest.json (see
 * scripts/genGetStartedMock.py and genGetStartedReal.py).
 */

// soften a detail inset's edges so it blends into the full slice
function featherEdges(img) {
  const w = img.naturalWidth;
  const h = img.naturalHeight;
  const c = document.createElement("canvas");
  c.width = w;
  c.height = h;
  const ctx = c.getContext("2d");
  ctx.drawImage(img, 0, 0);
  const F = Math.round(Math.min(w, h) * 0.04);
  ctx.globalCompositeOperation = "destination-out";
  for (const [x0, y0, x1, y1] of [
    [0, 0, F, 0],
    [w, 0, w - F, 0],
    [0, 0, 0, F],
    [0, h, 0, h - F],
  ]) {
    const g = ctx.createLinearGradient(x0, y0, x1, y1);
    g.addColorStop(0, "rgba(0,0,0,1)");
    g.addColorStop(1, "rgba(0,0,0,0)");
    ctx.fillStyle = g;
    ctx.fillRect(0, 0, w, h);
  }
  ctx.globalCompositeOperation = "source-over";
  return c;
}

const SIGMA = 60; // px of render-x influence per dot
const DONE_COVERAGE = 0.6; // per-band fraction of columns segmented well
const DARK = 12;
const PULSE_STEP = 0.05; // the marker leads the trace tip by this much arc

export default function SegDemo() {
  const base = useBaseUrl("/get-started/seg/");
  const [phase, setPhase] = useState("idle"); // idle|loading|play|done|error
  const [active, setActive] = useState(0);
  const [opened, setOpened] = useState(false); // a plane has been picked
  const [coverage, setCoverage] = useState([0, 0, 0]);
  // 0 = fully clean (not even the tutorial pulse); the default sits a
  // little to the right so the pulse guides first-timers out of the box,
  // and pushing further adds hint dots, then edges, then the full line
  const [hintLevel, setHintLevel] = useState(15);
  const [spiralOn, setSpiralOn] = useState(false);
  const [nextUp, setNextUp] = useState(null); // plane finished -> next one offered
  const [cueDismissed, setCueDismissed] = useState({}); // per-slice "keep tweaking"
  const [doneCueSeen, setDoneCueSeen] = useState(false); // phone finish pill tapped
  const [, setTick] = useState(0); // bumps when dots/zoom change
  const coarse = useCoarsePointer(); // gesture wording matches the device
  const [status, setStatus] = useState(
    "Tap the pulsing ring on the scroll to open that cross-section.",
  );
  // re-word the pre-baked intro status once the pointer type is known
  useEffect(() => {
    const intro = " the pulsing ring on the scroll to open that cross-section.";
    setStatus((s) => (s.endsWith(intro) ? (coarse ? "Tap" : "Click") + intro : s));
  }, [coarse]);

  const sliceRef = useRef(null);
  const renderRef = useRef(null);
  const ttRef = useRef(null);
  const outroRef = useRef(null); // done payoff, scrolled to from the phone pill
  const scrollPaneRef = useRef(null); // the turntable column, for guided hops
  const S = useRef(null);
  const activeRef = useRef(0);
  activeRef.current = active;
  const openedRef = useRef(opened);
  openedRef.current = opened;
  const phaseRef = useRef(phase);
  phaseRef.current = phase;
  const finishTest = useRef(false);
  const hintRef = useRef(hintLevel);
  hintRef.current = hintLevel;
  const drawSliceRef = useRef(() => {});
  const drawTTRef = useRef(() => {});

  async function start() {
    setPhase("loading");
    try {
      const m = await loadManifest(base);
      // the demo opens on slices + renders (~1.5MB); the 3.3MB turntable
      // sprite streams in behind it (the SVG schematic stands in until it
      // lands) and the ~0.4MB hi-res insets load per slice on first open —
      // blocking the button on all ~6MB made phones wait for nothing
      const [sliceImgs, renderImgs] = await Promise.all([
        Promise.all(m.slices.map((s) => loadImage(base + s.image))),
        Promise.all(m.render.images.map((n) => loadImage(base + n))),
      ]);
      const { width: RW, height: RH } = m.render;
      const renders = renderImgs.map((img) => imageToGray(img, RW, RH));
      S.current = {
        m,
        sliceImgs,
        // hi-res insets around the trace (feathered on arrival), fetched
        // lazily by ensureDetail the first time each slice opens
        detailCvs: m.slices.map(() => null),
        detailBusy: m.slices.map(() => false),
        renders,
        ttImg: null,
        RW,
        RH,
        dots: m.slices.map(() => []),
        views: m.slices.map(() => ({ s: 1, tx: 0, ty: 0 })),
        pointers: new Map(),
        gesture: null,
        downPt: null,
        viewStart: null,
        pinch0: null,
        out: null,
        cov: [0, 0, 0],
        prevCov: [0, 0, 0],
        doneShown: false,
        ttFrame: 0,
        ttTouched: false,
        ttDrag: null,
        ttView: { s: 1, tx: 0, ty: 0 }, // turntable zoom/pan
        ttPts: new Map(),
        ttPinch: null,
        glide: 0,
        autoTimer: 0,
      };
      const rc = renderRef.current;
      rc.width = RW; // native render aspect: same on-screen height as the
      rc.height = RH; // slice pane (the CSS grid columns carry the ratio)
      S.current.out = rc.getContext("2d").createImageData(RW, RH);
      setPhase("play");
      drawSlice(0);
      compositeRender();
      if (m.turntable)
        loadImage(base + m.turntable.sprite)
          .then((img) => {
            const st = S.current;
            if (!st) return;
            st.ttImg = img;
            drawTTRef.current();
            setTick((t) => t + 1); // hasTT flips: canvas replaces the SVG
          })
          .catch(() => {
            /* the SVG schematic keeps working without the sprite */
          });
    } catch (e) {
      console.error(e);
      setPhase("error");
    }
  }

  // hi-res inset for one slice, fetched the first time that slice opens:
  // each is ~0.4MB and only the planes actually traced get paid for. The
  // slice draws without it and the inset blends in when it lands.
  function ensureDetail(i) {
    const st = S.current;
    const d = st?.m.slices[i]?.detail;
    if (!st || !d || st.detailCvs[i] || st.detailBusy[i]) return;
    st.detailBusy[i] = true;
    loadImage(base + d.image)
      .then((img) => {
        const cur = S.current;
        if (!cur) return;
        cur.detailCvs[i] = featherEdges(img);
        if (activeRef.current === i) drawSliceRef.current(i);
      })
      .catch(() => {
        if (S.current) S.current.detailBusy[i] = false; // retry on reopen
      });
  }

  function selectSlice(i) {
    // sync the ref before drawing: drawTT reads activeRef, and the render
    // that refreshes it hasn't happened yet (this was the one-tap-behind bug)
    activeRef.current = i;
    setActive(i);
    ensureDetail(i);
    const st = S.current;
    const first = !openedRef.current;
    openedRef.current = true;
    setOpened(true);
    setNextUp(null);
    // single-column layouts put the slice pane below the scroll: bring
    // the place the user should click next onto the screen
    if (st && !st.autoTimer && window.innerWidth < 997)
      sliceRef.current?.scrollIntoView({ behavior: "smooth", block: "center" });
    // untouched slices open zoomed in on the start of the target sheet,
    // with the first dot already placed as a worked example (skipped
    // during the auto-finish, which places its own dots on a full view)
    if (st && !st.autoTimer && st.dots[i].length === 0) {
      introZoom(i);
      const s = st.m.slices[i];
      // seed the worked-example dot at the LEFT end (t = 1); the trace runs
      // right from there and the render fills in from its right side
      const [x0, y0] = s.surface[s.surface.length - 1];
      st.dots[i].push({ x: x0, y: y0, t: 1, q: 1 });
      setStatus(
        first
          ? "This is one slice through the roll, zoomed to your sheet. We placed the first dot. Tap the pulsing marker to extend the line."
          : "Same sheet, new height. Your first dot is placed. Keep going from the pulsing marker.",
      );
      compositeRender();
    }
    drawSlice(i);
    drawTT();
    setTick((t) => t + 1);
  }

  // zoom the view onto the start of the GT sheet, leaving room in the
  // direction the trace will grow
  function introZoom(i) {
    const st = S.current;
    const s = st.m.slices[i];
    const n = s.surface.length - 1;
    const [x0, y0] = s.surface[Math.round(0.97 * n)];
    const [x1] = s.surface[Math.round(0.7 * n)];
    const v = st.views[i];
    v.s = st.m.introZoom || 2.2;
    v.tx = s.width * (x1 > x0 ? 0.3 : 0.7) - x0 * v.s;
    v.ty = s.height * 0.5 - y0 * v.s;
    clampView(v, s.width, s.height);
  }

  // ---- view (zoom/pan) helpers -------------------------------------------

  function clampView(v, W, H) {
    const mz = S.current?.m.maxZoom || 6;
    v.s = Math.min(mz, Math.max(1, v.s));
    v.tx = Math.min(0, Math.max(W - W * v.s, v.tx));
    v.ty = Math.min(0, Math.max(H - H * v.s, v.ty));
  }

  // non-passive wheel listener (React's onWheel can't preventDefault)
  useEffect(() => {
    const c = sliceRef.current;
    if (!c) return;
    const onWheel = (e) => {
      const st = S.current;
      if (!st || !openedRef.current) return;
      e.preventDefault();
      if (st.glide) {
        cancelAnimationFrame(st.glide);
        st.glide = 0;
      }
      const i = activeRef.current;
      const v = st.views[i];
      const rect = c.getBoundingClientRect();
      const cx = ((e.clientX - rect.left) / rect.width) * c.width;
      const cy = ((e.clientY - rect.top) / rect.height) * c.height;
      const ix = (cx - v.tx) / v.s;
      const iy = (cy - v.ty) / v.s;
      // free-spinning wheels deliver single events with |deltaY| in the
      // thousands — unclamped, one reverse-momentum tick slams the view
      // from max zoom to minimum. Cap each event at ~2x.
      v.s = v.s * Math.pow(1.0018, -Math.max(-400, Math.min(400, e.deltaY)));
      v.tx = cx - ix * v.s;
      v.ty = cy - iy * v.s;
      clampView(v, c.width, c.height);
      drawSliceRef.current(i);
      setTick((t) => t + 1);
    };
    c.addEventListener("wheel", onWheel, { passive: false });
    return () => c.removeEventListener("wheel", onWheel);
  }, []);

  // ---- middle pane: cross-section ----------------------------------------

  function drawSlice(i) {
    const st = S.current;
    if (!st || !sliceRef.current) return;
    const s = st.m.slices[i];
    const c = sliceRef.current;
    c.width = s.width; // also resets the transform
    c.height = s.height;
    const ctx = c.getContext("2d");
    c.dataset.opened = openedRef.current ? "1" : "0";
    if (!openedRef.current) {
      // nothing to trace yet: the scroll pane leads
      const f1 = Math.round(c.width / 25);
      const f2 = Math.round(c.width / 32);
      ctx.fillStyle = "#0b0c0e";
      ctx.fillRect(0, 0, c.width, c.height);
      ctx.textAlign = "center";
      ctx.fillStyle = "rgba(185,180,174,0.85)";
      ctx.font = `600 ${f1}px system-ui, sans-serif`;
      ctx.fillText("Pick a plane on the scroll", c.width / 2, c.height / 2 - f2);
      ctx.fillStyle = "rgba(138,133,127,0.95)";
      ctx.font = `${f2}px system-ui, sans-serif`;
      ctx.fillText(
        "Its cross-section opens here",
        c.width / 2,
        c.height / 2 + f2 * 1.2,
      );
      delete c.dataset.pulse;
      return;
    }
    const v = st.views[i];
    c.dataset.view = `${v.s.toFixed(3)},${v.tx.toFixed(1)},${v.ty.toFixed(1)}`;
    ctx.setTransform(v.s, 0, 0, v.s, v.tx, v.ty);
    ctx.drawImage(st.sliceImgs[i], 0, 0);
    // hi-res inset over the trace region (feathered edges)
    const det = st.m.slices[i].detail;
    if (det && st.detailCvs[i])
      ctx.drawImage(st.detailCvs[i], det.x, det.y, det.side, det.side);
    drawHints(ctx, s, v.s);

    const lw = 2 / Math.sqrt(v.s);
    const rDot = st.m.dotRadius / Math.sqrt(v.s);

    // connect the dots in trace (arc) order — the surface line forming
    const sorted = [...st.dots[i]].sort((a, b) => a.t - b.t);
    for (let k = 0; k + 1 < sorted.length; k++) {
      const a = sorted[k];
      const b = sorted[k + 1];
      ctx.beginPath();
      ctx.moveTo(a.x, a.y);
      ctx.lineTo(b.x, b.y);
      ctx.strokeStyle =
        Math.min(a.q, b.q) > 0.6
          ? "rgba(229,80,43,0.85)"
          : "rgba(160,160,160,0.6)";
      ctx.lineWidth = lw;
      ctx.stroke();
    }

    // dots
    for (const d of st.dots[i]) {
      ctx.beginPath();
      ctx.arc(d.x, d.y, rDot, 0, Math.PI * 2);
      ctx.fillStyle =
        d.q > 0.6 ? "rgba(229,80,43,0.9)" : "rgba(160,160,160,0.8)";
      ctx.fill();
      ctx.strokeStyle = "rgba(255,255,255,0.7)";
      ctx.lineWidth = lw * 0.75;
      ctx.stroke();
    }

    // pulsing "continue here" marker just past the trace tip: it leads the
    // user along the one winding we want, past all the look-alike wraps
    const pp =
      phaseRef.current === "play" && hintRef.current > 0 ? pulsePoint(i) : null;
    c.dataset.pulse = pp ? `${Math.round(pp.x)},${Math.round(pp.y)}` : "";
    if (pp) {
      const ph = performance.now() / 550;
      ctx.beginPath();
      ctx.arc(pp.x, pp.y, (12 + 3 * Math.sin(ph)) / Math.sqrt(v.s), 0, Math.PI * 2);
      ctx.strokeStyle = `rgba(255,106,61,${0.85 - 0.25 * Math.sin(ph)})`;
      ctx.lineWidth = 2.6 / Math.sqrt(v.s);
      ctx.stroke();
      ctx.beginPath();
      ctx.arc(pp.x, pp.y, 3 / Math.sqrt(v.s), 0, Math.PI * 2);
      ctx.fillStyle = "rgba(255,106,61,0.9)";
      ctx.fill();
      // first-time affordance: label the marker until the user has placed
      // a dot of their own on this slice (the seed dot doesn't count)
      if (st.dots[i].length < 2 && !st.autoTimer) {
        const k = c.width / (c.getBoundingClientRect().width || c.width);
        drawPill(ctx, pillLabel(), pp.x, pp.y, 15 / Math.sqrt(v.s), k / v.s, s.width);
      }
    }
    ctx.setTransform(1, 0, 0, 1, 0, 0);
  }
  drawSliceRef.current = drawSlice;

  // the next step along the sheet: just past the furthest good dot. The
  // trace runs from the LEFT end of the slice (t = 1, the easy part) toward
  // the right (t = 0), so the flattened sheet fills in from its right side.
  // Shown whenever Hints is above 0 (slider hard left = fully clean), and
  // it keeps pulsing at the far end until the plane is actually done.
  function pulsePoint(i) {
    const st = S.current;
    if (!st || sliceReachedEnd(i)) return null;
    const s = st.m.slices[i];
    let tMin = 1 + PULSE_STEP;
    for (const d of st.dots[i]) if (d.q > 0.6 && d.t < tMin) tMin = d.t;
    const target = Math.max(0.01, Math.min(1, tMin - PULSE_STEP));
    const idx = Math.round(target * (s.surface.length - 1));
    const [x, y] = s.surface[idx];
    return { x, y };
  }

  // Coverage measures how much of the flattened band looks usable, but it
  // is not the end of the guided trace. Keep guiding until a good dot lands
  // at the actual far end of this tifxyz intersection.
  function sliceReachedEnd(i) {
    const st = S.current;
    return !!st?.dots[i]?.some((d) => d.q > 0.6 && d.t <= 0.015);
  }

  // glide the zoomed view so the marker stays on screen as the trace grows
  function followMarker(i) {
    const st = S.current;
    const c = sliceRef.current;
    const v = st.views[i];
    if (v.s <= 1.2) return;
    const pp = pulsePoint(i);
    if (!pp) return;
    const sx = pp.x * v.s + v.tx;
    const sy = pp.y * v.s + v.ty;
    if (
      sx > c.width * 0.15 &&
      sx < c.width * 0.85 &&
      sy > c.height * 0.15 &&
      sy < c.height * 0.85
    )
      return;
    const goal = { s: v.s, tx: c.width / 2 - pp.x * v.s, ty: c.height / 2 - pp.y * v.s };
    clampView(goal, c.width, c.height);
    if (window.matchMedia?.("(prefers-reduced-motion: reduce)").matches) {
      v.tx = goal.tx;
      v.ty = goal.ty;
      drawSlice(i);
      return;
    }
    const from = { tx: v.tx, ty: v.ty };
    const t0 = performance.now();
    if (st.glide) cancelAnimationFrame(st.glide);
    const step = () => {
      if (!S.current) return;
      const k = Math.min(1, (performance.now() - t0) / 340);
      const e = k * (2 - k);
      v.tx = from.tx + (goal.tx - from.tx) * e;
      v.ty = from.ty + (goal.ty - from.ty) * e;
      drawSliceRef.current(i);
      st.glide = k < 1 ? requestAnimationFrame(step) : 0;
    };
    st.glide = requestAnimationFrame(step);
  }

  // keep the markers breathing (skipped under prefers-reduced-motion)
  useEffect(() => {
    if (phase !== "play") return;
    if (window.matchMedia?.("(prefers-reduced-motion: reduce)").matches)
      return;
    const id = setInterval(() => {
      drawSliceRef.current(activeRef.current);
      drawTTRef.current(); // the plane-picker marker pulses too
    }, 90);
    return () => clearInterval(id);
  }, [phase]);

  // the Hints slider: 0 = clean, the first stretch = just the tutorial
  // pulse (the default), then a few dots -> more dots -> partial
  // connections -> the full ground-truth line
  function drawHints(ctx, s, scale) {
    const h = Math.max(0, (hintRef.current / 100 - 0.25) / 0.75);
    if (h <= 0.01) return;
    const pts = s.surface;
    ctx.save();

    // start/end markers
    const [sx, sy] = pts[0];
    const [ex, ey] = pts[pts.length - 1];
    ctx.fillStyle = "rgba(255,106,61,0.7)";
    ctx.beginPath();
    ctx.arc(sx, sy, 6 / Math.sqrt(scale), 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = "rgba(255,106,61,0.7)";
    ctx.lineWidth = 1.5 / Math.sqrt(scale);
    ctx.beginPath();
    ctx.arc(ex, ey, 6 / Math.sqrt(scale), 0, Math.PI * 2);
    ctx.stroke();

    const n = Math.max(2, Math.round(2 + h * 26));
    const idxs = Array.from({ length: n }, (_, k) =>
      Math.round((k * (pts.length - 1)) / (n - 1)),
    );

    ctx.strokeStyle = "rgba(255,106,61,0.5)";
    ctx.lineWidth = 3.5 / Math.sqrt(scale);
    ctx.setLineDash([7 / scale, 7 / scale]);
    if (h >= 0.92) {
      // full ground-truth line
      ctx.beginPath();
      pts.forEach(([x, y], k) => (k === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y)));
      ctx.stroke();
    } else if (h > 0.55) {
      // connect a growing fraction of the gaps between hint dots
      const frac = (h - 0.55) / 0.37;
      for (let k = 0; k + 1 < idxs.length; k++) {
        if (((k * 0.618034) % 1) >= frac) continue;
        ctx.beginPath();
        for (let j = idxs[k]; j <= idxs[k + 1]; j++) {
          const [x, y] = pts[j];
          j === idxs[k] ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
        }
        ctx.stroke();
      }
    }
    ctx.setLineDash([]);

    // hint dots: a brighter orange at lower opacity reads as a guide
    // without covering the data underneath
    ctx.fillStyle = "rgba(255,106,61,0.45)";
    ctx.strokeStyle = "rgba(255,106,61,0.65)";
    ctx.lineWidth = 1.2 / Math.sqrt(scale);
    for (const k of idxs) {
      const [x, y] = pts[k];
      ctx.beginPath();
      ctx.arc(x, y, 5.5 / Math.sqrt(scale), 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();
    }
    ctx.restore();
  }

  // pointer gestures on the slice: tap = dot, drag = pan (when zoomed),
  // two pointers = pinch zoom
  function slicePt(e) {
    const [x, y] = eventToImageXY(e, sliceRef.current);
    return { x, y };
  }

  function onSliceDown(e) {
    if (phase !== "play" && phase !== "done") return;
    if (!openedRef.current) return; // pick a plane on the scroll first
    const st = S.current;
    if (st.glide) {
      cancelAnimationFrame(st.glide);
      st.glide = 0;
    }
    try {
      e.currentTarget.setPointerCapture(e.pointerId);
    } catch {
      /* pointer already gone — capture is best-effort */
    }
    st.pointers.set(e.pointerId, slicePt(e));
    if (e.pointerType === "mouse" && e.button !== 0) {
      // right/middle drag pans without the tap-vs-pan dance (and can
      // never drop a dot by accident)
      e.preventDefault();
      st.gesture = "pan";
      st.downPt = slicePt(e);
      st.viewStart = { ...st.views[activeRef.current] };
      return;
    }
    if (st.pointers.size === 2) {
      const [p1, p2] = [...st.pointers.values()];
      const v = st.views[activeRef.current];
      const mid = { x: (p1.x + p2.x) / 2, y: (p1.y + p2.y) / 2 };
      st.pinch0 = {
        d: Math.max(1, Math.hypot(p1.x - p2.x, p1.y - p2.y)),
        s: v.s,
        ix: (mid.x - v.tx) / v.s,
        iy: (mid.y - v.ty) / v.s,
      };
      st.gesture = "pinch";
    } else {
      st.gesture = "maybe-tap";
      st.downPt = slicePt(e);
      st.viewStart = { ...st.views[activeRef.current] };
    }
  }

  function onSliceMove(e) {
    const st = S.current;
    if (!st || !st.pointers.has(e.pointerId)) return;
    st.pointers.set(e.pointerId, slicePt(e));
    const c = sliceRef.current;
    if (st.gesture === "pinch" && st.pointers.size === 2) {
      const [p1, p2] = [...st.pointers.values()];
      const d = Math.max(1, Math.hypot(p1.x - p2.x, p1.y - p2.y));
      const mid = { x: (p1.x + p2.x) / 2, y: (p1.y + p2.y) / 2 };
      const v = st.views[activeRef.current];
      v.s = st.pinch0.s * (d / st.pinch0.d);
      v.tx = mid.x - st.pinch0.ix * v.s;
      v.ty = mid.y - st.pinch0.iy * v.s;
      clampView(v, c.width, c.height);
      drawSlice(activeRef.current);
    } else if (st.gesture === "maybe-tap" || st.gesture === "pan") {
      const p = slicePt(e);
      const moved = Math.hypot(p.x - st.downPt.x, p.y - st.downPt.y);
      if (st.gesture === "maybe-tap" && moved > 8) st.gesture = "pan";
      if (st.gesture === "pan") {
        const v = st.views[activeRef.current];
        if (v.s > 1) {
          v.tx = st.viewStart.tx + (p.x - st.downPt.x) * 1; // canvas px
          v.ty = st.viewStart.ty + (p.y - st.downPt.y) * 1;
          clampView(v, c.width, c.height);
          drawSlice(activeRef.current);
        }
      }
    }
  }

  function onSliceUp(e) {
    const st = S.current;
    if (!st) return;
    const wasTap = st.gesture === "maybe-tap";
    st.pointers.delete(e.pointerId);
    if (st.pointers.size === 0) {
      const g = st.gesture;
      st.gesture = null;
      if (wasTap && g === "maybe-tap") tapAt(st.downPt);
      setTick((t) => t + 1);
    }
  }

  function tapAt(pt) {
    const st = S.current;
    const i = activeRef.current;
    const s = st.m.slices[i];
    const v = st.views[i];
    const x = (pt.x - v.tx) / v.s;
    const y = (pt.y - v.ty) / v.s;
    if (x < 0 || y < 0 || x >= s.width || y >= s.height) return;

    // tap an existing dot -> remove it
    const hitR = (st.m.dotRadius * 2.2) / Math.sqrt(v.s);
    const hit = st.dots[i].findIndex(
      (d) => (d.x - x) ** 2 + (d.y - y) ** 2 < hitR * hitR,
    );
    if (hit >= 0) {
      st.dots[i].splice(hit, 1);
    } else {
      // nearest point on the GT polyline
      let bd = Infinity;
      let bk = 0;
      s.surface.forEach(([px, py], k) => {
        const d2 = (px - x) ** 2 + (py - y) ** 2;
        if (d2 < bd) {
          bd = d2;
          bk = k;
        }
      });
      const dist = Math.sqrt(bd);
      const t = bk / (s.surface.length - 1);
      const { tolerancePx: tol, maxDistPx: max } = st.m;
      const q = Math.max(0, Math.min(1, 1 - (dist - tol) / (max - tol)));
      st.dots[i].push({ x, y, t, q });
      if (q > 0.85) setStatus("");
      else if (q > 0.3)
        setStatus(
          "Close, but slightly off the sheet. See the smearing on the right?",
        );
      else
        setStatus(
          "That dot landed on a neighboring layer, or in the gap between layers. The render on the right goes wrong there.",
        );
      if (q > 0.6) followMarker(i); // keep the next step on screen
    }
    drawSlice(i);
    compositeRender();
    drawTT();
    setTick((t) => t + 1);
  }

  // hands-off ending for visitors who've dropped a few dots and want to
  // see where it goes: dots walk the true sheet, plane by plane
  function finishForMe() {
    const st = S.current;
    if (!st || st.autoTimer) return;
    setStatus("Watch: dots following the same sheet on all three planes.");
    // frame each view on the trace region; the full slice is too wide to
    // watch the auto-trace on
    st.views = st.m.slices.map((s) => {
      const xs = s.surface.map((p) => p[0]);
      const ys = s.surface.map((p) => p[1]);
      const bw = Math.max(1, Math.max(...xs) - Math.min(...xs));
      const bh = Math.max(1, Math.max(...ys) - Math.min(...ys));
      const zoom = Math.max(
        1,
        Math.min(st.m.maxZoom || 6, 0.7 * Math.min(s.width / bw, s.height / bh)),
      );
      const v = {
        s: zoom,
        tx: s.width / 2 - ((Math.min(...xs) + Math.max(...xs)) / 2) * zoom,
        ty: s.height / 2 - ((Math.min(...ys) + Math.max(...ys)) / 2) * zoom,
      };
      clampView(v, s.width, s.height);
      return v;
    });
    const plan = [];
    st.m.slices.forEach((s, si) => {
      // walk each slice in the real trace direction, t = 1 -> 0: the
      // t <= 0.015 "slice done" dot lands LAST, so every plane gets its
      // full run of dots (placing t=0 first ended the demo after one dot
      // per plane, with most of the surface never traced)
      const n = 12;
      for (let k = 0; k < n; k++) {
        const idx = Math.round(((n - 1 - k) * (s.surface.length - 1)) / (n - 1));
        const [x, y] = s.surface[idx];
        plan.push({ si, x, y, t: idx / (s.surface.length - 1) });
      }
    });
    let j = 0;
    st.autoTimer = setInterval(() => {
      const cur = S.current;
      if (!cur) return;
      if (j >= plan.length || cur.doneShown) {
        clearInterval(cur.autoTimer);
        cur.autoTimer = 0;
        setTick((t) => t + 1);
        return;
      }
      const p = plan[j++];
      if (activeRef.current !== p.si) selectSlice(p.si);
      cur.dots[p.si].push({ x: p.x, y: p.y, t: p.t, q: 1 });
      drawSlice(p.si);
      compositeRender();
      drawTT();
      setTick((t) => t + 1);
    }, 110);
  }

  useEffect(
    () => () => {
      if (S.current?.autoTimer) clearInterval(S.current.autoTimer);
    },
    [],
  );

  // idle prefetch: when the demo scrolls near on a connection that can
  // afford it, warm the HTTP cache for the big assets so "Trace the
  // sheet" starts instantly. Click-gating stays the source of truth —
  // start() simply finds everything already local. Skipped on save-data
  // and slow links, where paying ~5MB uninvited would be rude.
  useEffect(() => {
    if (phase !== "idle") return;
    const conn = navigator.connection;
    if (conn && (conn.saveData || /(^|\b)[23]g$/.test(conn.effectiveType || "")))
      return;
    const el = document.getElementById("seg-demo");
    if (!el || typeof IntersectionObserver === "undefined") return;
    let fired = false;
    const io = new IntersectionObserver(
      (entries) => {
        if (fired || !entries.some((e) => e.isIntersecting)) return;
        fired = true;
        io.disconnect();
        const idle = window.requestIdleCallback || ((f) => setTimeout(f, 700));
        idle(async () => {
          try {
            const m = await loadManifest(base);
            for (const n of [
              ...m.slices.map((s) => s.image),
              ...m.render.images,
              m.turntable?.sprite,
            ]) {
              if (!n) continue;
              const img = new Image();
              img.src = base + n;
            }
          } catch {
            /* prefetch is best-effort */
          }
        });
      },
      { rootMargin: "600px" },
    );
    io.observe(el);
    return () => io.disconnect();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [phase]);

  // test shortcut: /get_started?finish=1 runs the demo to completion and
  // opens the spiral, so the end states can be reviewed without playing
  useEffect(() => {
    if (!new URLSearchParams(window.location.search).has("finish")) return;
    finishTest.current = true;
    start().then(() => setTimeout(finishForMe, 400));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);
  useEffect(() => {
    if (phase === "done" && finishTest.current) setSpiralOn(true);
  }, [phase]);

  // "trace the next layer" POINTS at the scroll instead of switching for
  // you: the ring for the missing plane is already pulsing there (with
  // its tap-here pill back on), and picking the plane yourself is the
  // lesson — slices live inside a volume. Tapping the ring then opens
  // the slice view exactly as before.
  function goPickNextPlane() {
    setCueDismissed((c) => ({ ...c, [activeRef.current]: true }));
    setStatus(
      (coarse ? "Tap" : "Click") +
        " the pulsing ring on the scroll to open the next plane.",
    );
    scrollPaneRef.current?.scrollIntoView({
      behavior: "smooth",
      block: "center",
    });
    drawTTRef.current(); // the pill re-appears on the ring right away
  }

  function clearSlice() {
    const st = S.current;
    st.dots[activeRef.current] = [];
    // retracing this slice to the end should announce itself again
    setCueDismissed((c) => ({ ...c, [activeRef.current]: false }));
    drawSlice(activeRef.current);
    compositeRender();
    drawTT();
    setTick((t) => t + 1);
    setStatus("Cleared. Start the line again from the pulsing marker.");
  }

  // ---- right pane: flattened render composite ----------------------------

  function compositeRender() {
    const st = S.current;
    if (!st || !renderRef.current) return;
    const { m, renders, RW, RH, out } = st;
    const K = renders.length;
    const cov = [0, 0, 0];

    for (let si = 0; si < m.slices.length; si++) {
      const s = m.slices[si];
      const [y0, y1] = s.vBand;
      // per-column quality & coverage from this slice's dots. Quality is
      // the weight-averaged q of every dot influencing the column — a dot
      // on the wrong layer smears the render around it even when good
      // dots sit nearby (the surface fit has to honor ALL the points),
      // and removing the bad dot heals it.
      const Qs = new Float32Array(RW);
      const Ws = new Float32Array(RW);
      const C = new Float32Array(RW);
      for (const d of st.dots[si]) {
        const cx = d.t * RW;
        const lo = Math.max(0, Math.floor(cx - 3 * SIGMA));
        const hi = Math.min(RW - 1, Math.ceil(cx + 3 * SIGMA));
        for (let x = lo; x <= hi; x++) {
          const w = Math.exp(-((x - cx) ** 2) / (2 * SIGMA * SIGMA));
          if (w > C[x]) C[x] = w;
          Qs[x] += d.q * w;
          Ws[x] += w;
        }
      }
      const Q = new Float32Array(RW);
      let good = 0;
      for (let x = 0; x < RW; x++) {
        Q[x] = Ws[x] > 0 ? Qs[x] / Ws[x] : 0;
        if (C[x] > 0.5 && Q[x] > 0.5) good++;
      }
      cov[si] = good / RW;

      for (let y = y0; y < y1; y++) {
        for (let x = 0; x < RW; x++) {
          const p = y * RW + x;
          // quality picks the normal-offset render (1 -> crisp, 0 -> worst)
          const fi = (1 - Q[x]) * (K - 1);
          const i0 = Math.min(K - 2, Math.floor(fi));
          const f = fi - i0;
          const v = renders[i0][p] * (1 - f) + renders[i0 + 1][p] * f;
          const shown = DARK + (v - DARK) * C[x];
          out.data[p * 4] = shown;
          out.data[p * 4 + 1] = shown;
          out.data[p * 4 + 2] = shown;
          out.data[p * 4 + 3] = 255;
        }
      }
    }
    // Congratulate when the trace reaches the real end of a slice, and
    // offer the next plane as a one-tap button (the scroll keeps pulsing
    // for people who'd rather pick it there).
    m.slices.forEach((_, i) => {
      if (
        sliceReachedEnd(i) &&
        !st.reachedEnd?.[i] &&
        !m.slices.every((__, j) => sliceReachedEnd(j))
      ) {
        setStatus(`Nice tracing. The ${m.slices[i].label} plane is done.`);
        if (!st.autoTimer)
          setNextUp(m.slices.findIndex((__, j) => !sliceReachedEnd(j)));
      }
    });
    st.reachedEnd = m.slices.map((_, i) => sliceReachedEnd(i));
    st.prevCov = cov;
    st.cov = cov;
    setCoverage(cov);

    const rctx = renderRef.current.getContext("2d");
    rctx.putImageData(out, 0, 0);
    if (st.dots.every((d) => d.length === 0) && !st.doneShown) {
      rctx.textAlign = "center";
      rctx.fillStyle = "rgba(138,133,127,0.9)";
      rctx.font = "600 24px system-ui, sans-serif";
      rctx.fillText(
        "The flattened sheet appears here as you trace",
        RW / 2,
        RH / 2,
      );
    }

    // the render stays a live function of the dots — before AND after
    // done, like the ink demo: drop a bad dot on a finished sheet and it
    // smears, remove it and it heals. Done itself unwinds if the trace
    // stops reaching the end (clearing a slice, deleting the end dot).
    const allDone = m.slices.every((_, i) => sliceReachedEnd(i));
    if (allDone && !st.doneShown) {
      st.doneShown = true;
      setPhase("done");
      setStatus("");
      drawTT();
    } else if (!allDone && st.doneShown) {
      st.doneShown = false;
      setPhase("play");
      setDoneCueSeen(false); // re-finishing should announce itself again
      drawTT();
    }
  }

  // ---- left pane: textured turntable of the real segment ------------------

  // which plane the turntable marker should pulse on: the first incomplete
  // one, but only while the user isn't mid-trace on the open slice. Hidden
  // when the Hints slider sits hard left.
  function ttPulseTarget() {
    const st = S.current;
    if (!st || phaseRef.current !== "play" || hintRef.current <= 0)
      return null;
    const next = st.m.slices.findIndex((_, i) => !sliceReachedEnd(i));
    if (next < 0) return null;
    if (!openedRef.current) return next;
    return sliceReachedEnd(activeRef.current) ? next : null;
  }

  function drawTT() {
    const st = S.current;
    const c = ttRef.current;
    if (!st || !st.ttImg || !c) return;
    const tt = st.m.turntable;
    const { tileW, tileH, cols, frames } = tt;
    c.width = tileW;
    c.height = tileH;
    c.dataset.drawnPlane = String(activeRef.current); // testable draw state
    const ctx = c.getContext("2d");
    // zoom/pan: everything below draws in tile coordinates under this
    // transform; UI strokes divide by v.s to keep constant screen size
    const v = st.ttView;
    ctx.setTransform(v.s, 0, 0, v.s, v.tx, v.ty);
    c.dataset.ttview = `${v.s.toFixed(3)},${v.tx.toFixed(1)},${v.ty.toFixed(1)}`;
    // UI sizes are authored for the original 480px tile; uz keeps them the
    // same on-screen size for any sprite resolution
    const uz = tileW / 480;
    const f = ((st.ttFrame % frames) + frames) % frames;

    // phantom slice planes through the volume: the far half is drawn
    // first (behind the sheet), the near half after the sprite tile
    const pcx = tileW / 2;
    const prx = tt.planeRx || tileW / 2 - 8;
    const pry = Math.max(4, tt.planeRy || 10);
    // the rings spin WITH the scroll: an ellipse is rotationally symmetric,
    // so the rotation is carried by a dash pattern whose offset slides
    // around the ring with the sprite frame. The offset advances a fixed
    // 30% of the dash period per frame — tying it to the true arc length
    // (circumference/frames) aliased to EXACTLY one period and froze the
    // dashes. Negative offset moves near-arc dashes the same way the near
    // surface moves; 0.3*frames = 12 full periods, so the loop is seamless.
    const ringDash = [16 * uz, 9 * uz];
    const ringSpin = -f * (ringDash[0] + ringDash[1]) * 0.3;
    // the orange emphasis follows the FLOW, not just the open slice:
    // while the user is being asked to hop planes (pulse target set), it
    // moves to the target ring — an orange highlight lingering on the
    // plane they just finished read as "click this one" and pointed the
    // wrong way from the pulse
    const hop = ttPulseTarget();
    const emph = hop != null ? hop : activeRef.current;
    c.dataset.emphPlane = String(emph);
    tt.planes.forEach((p, i) => {
      const on = emph === i;
      if (on) {
        ctx.fillStyle = "rgba(229,80,43,0.10)";
        ctx.beginPath();
        ctx.ellipse(pcx, p.screenY, prx, pry, 0, 0, Math.PI * 2);
        ctx.fill();
      }
      ctx.strokeStyle = on
        ? "rgba(229,80,43,0.45)"
        : "rgba(255,255,255,0.16)";
      ctx.lineWidth = uz;
      ctx.setLineDash(ringDash);
      ctx.lineDashOffset = ringSpin;
      ctx.beginPath();
      ctx.ellipse(pcx, p.screenY, prx, pry, 0, Math.PI, Math.PI * 2);
      ctx.stroke();
      ctx.setLineDash([]);
    });

    ctx.drawImage(
      st.ttImg,
      (f % cols) * tileW,
      Math.floor(f / cols) * tileH,
      tileW,
      tileH,
      0,
      0,
      tileW,
      tileH,
    );

    // near half of the phantom planes + labels (tap targets). The three
    // real z-planes are close in this crop, so stagger only their labels;
    // the ring geometry itself stays at its measured position.
    ctx.font = `${10 * uz}px system-ui, sans-serif`;
    const labelY = tt.planes.map((p, i) => p.screenY + (i - 1) * 12 * uz);
    tt.planes.forEach((p, i) => {
      const on = emph === i && (openedRef.current || hop != null);
      ctx.strokeStyle = on ? "#E5502B" : "rgba(255,255,255,0.38)";
      ctx.lineWidth = (on ? 1.6 : 1) * uz;
      ctx.setLineDash(ringDash);
      ctx.lineDashOffset = ringSpin;
      ctx.beginPath();
      ctx.ellipse(pcx, p.screenY, prx, pry, 0, 0, Math.PI);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.strokeStyle = on ? "rgba(229,80,43,0.7)" : "rgba(255,255,255,0.28)";
      ctx.lineWidth = uz;
      ctx.beginPath();
      ctx.moveTo(34 * uz, labelY[i] - 3 * uz);
      ctx.lineTo(Math.max(4 * uz, pcx - prx), p.screenY);
      ctx.stroke();
      ctx.fillStyle = on ? "#E5502B" : "rgba(255,255,255,0.72)";
      ctx.fillText(st.m.slices[i].label, 4 * uz, labelY[i]);
      if (sliceReachedEnd(i)) {
        ctx.fillStyle = "#E5502B";
        ctx.fillText("✓", tileW - 14 * uz, p.screenY - 4 * uz);
      }
    });

    // pulsing "open this plane" marker: leads the whole flow — the first
    // pick, and the hop to the next plane once a band is traced (same
    // target the emphasis above follows)
    const tp = hop;
    c.dataset.pulsePlane = tp == null ? "" : String(tp);
    if (tp != null) {
      const p = tt.planes[tp];
      const ph = performance.now() / 550;
      ctx.beginPath();
      ctx.arc(pcx, p.screenY, (10 + 3 * Math.sin(ph)) * uz, 0, Math.PI * 2);
      ctx.strokeStyle = `rgba(255,106,61,${0.9 - 0.25 * Math.sin(ph)})`;
      ctx.lineWidth = (2.2 * uz) / v.s;
      ctx.stroke();
      ctx.beginPath();
      ctx.arc(pcx, p.screenY, 3 * uz, 0, Math.PI * 2);
      ctx.fillStyle = "rgba(255,106,61,0.95)";
      ctx.fill();
      // affordance pill: on the first pick, and again whenever the flow
      // asks for a plane hop (the active slice is traced) — the bare ring
      // alone didn't read as "click me" to people mid-flow
      if (
        (!openedRef.current || sliceReachedEnd(activeRef.current)) &&
        !st.autoTimer
      ) {
        const k = c.width / (c.getBoundingClientRect().width || c.width);
        drawPill(ctx, pillLabel(), pcx, p.screenY, 14 * uz, k / v.s, tileW);
      }
    }
    ctx.setTransform(1, 0, 0, 1, 0, 0);
  }
  drawTTRef.current = drawTT;

  // first paint of the turntable (its canvas mounts on the play re-render)
  useEffect(() => {
    if (phase === "play" || phase === "done") drawTTRef.current();
  }, [phase]);

  // gentle auto-rotate until the first touch
  useEffect(() => {
    if (phase !== "play") return;
    const st = S.current;
    if (!st?.ttImg) return;
    if (window.matchMedia?.("(prefers-reduced-motion: reduce)").matches)
      return;
    const id = setInterval(() => {
      const cur = S.current;
      if (!cur || cur.ttTouched) return;
      const n = cur.m.turntable.frames;
      cur.ttFrame = (cur.ttFrame - 1 + n) % n;
      drawTTRef.current();
    }, 140);
    return () => clearInterval(id);
  }, [phase]);

  function clampTTView(v) {
    const tt = S.current.m.turntable;
    v.s = Math.min(2.6, Math.max(1, v.s));
    v.tx = Math.min(0, Math.max(tt.tileW - tt.tileW * v.s, v.tx));
    v.ty = Math.min(0, Math.max(tt.tileH - tt.tileH * v.s, v.ty));
  }

  // wheel zoom on the turntable (non-passive; attaches once the canvas
  // exists, which is only after start())
  useEffect(() => {
    const c = ttRef.current;
    if (!c) return;
    const onWheel = (e) => {
      const st = S.current;
      if (!st?.ttImg) return;
      e.preventDefault();
      const v = st.ttView;
      const rect = c.getBoundingClientRect();
      const cx = ((e.clientX - rect.left) / rect.width) * c.width;
      const cy = ((e.clientY - rect.top) / rect.height) * c.height;
      const ix = (cx - v.tx) / v.s;
      const iy = (cy - v.ty) / v.s;
      // same per-event cap as the slice pane (free-spinning wheels)
      v.s = v.s * Math.pow(1.0018, -Math.max(-400, Math.min(400, e.deltaY)));
      v.tx = cx - ix * v.s;
      v.ty = cy - iy * v.s;
      clampTTView(v);
      drawTTRef.current();
    };
    c.addEventListener("wheel", onWheel, { passive: false });
    return () => c.removeEventListener("wheel", onWheel);
  }, [phase]);

  function onTTDown(e) {
    const st = S.current;
    if (!st || !st.ttImg) return;
    st.ttTouched = true;
    try {
      e.currentTarget.setPointerCapture(e.pointerId);
    } catch {
      /* pointer already gone — capture is best-effort */
    }
    const [cx, cy] = eventToImageXY(e, ttRef.current);
    if (e.pointerType === "mouse" && e.button !== 0) {
      // right/middle drag pans freely when zoomed (left drag keeps
      // spinning horizontally, so it can never reach the sides)
      e.preventDefault();
      st.ttPan = { x: cx, y: cy };
      return;
    }
    st.ttPts.set(e.pointerId, { x: cx, y: cy });
    if (st.ttPts.size === 2) {
      // pinch zoom
      st.ttDrag = null;
      const [p1, p2] = [...st.ttPts.values()];
      const v = st.ttView;
      const mid = { x: (p1.x + p2.x) / 2, y: (p1.y + p2.y) / 2 };
      st.ttPinch = {
        d: Math.max(1, Math.hypot(p1.x - p2.x, p1.y - p2.y)),
        s: v.s,
        ix: (mid.x - v.tx) / v.s,
        iy: (mid.y - v.ty) / v.s,
      };
    } else {
      st.ttDrag = {
        x: e.clientX,
        y: e.clientY,
        frame: st.ttFrame,
        tx: st.ttView.tx,
        ty: st.ttView.ty,
        moved: false,
      };
    }
  }
  function onTTMove(e) {
    const st = S.current;
    if (!st) return;
    if (st.ttPan) {
      const [cx, cy] = eventToImageXY(e, ttRef.current);
      const v = st.ttView;
      v.tx += cx - st.ttPan.x;
      v.ty += cy - st.ttPan.y;
      st.ttPan = { x: cx, y: cy };
      clampTTView(v);
      drawTT();
      return;
    }
    if (st.ttPts.has(e.pointerId)) {
      const [cx, cy] = eventToImageXY(e, ttRef.current);
      st.ttPts.set(e.pointerId, { x: cx, y: cy });
    }
    if (st.ttPinch && st.ttPts.size === 2) {
      const [p1, p2] = [...st.ttPts.values()];
      const d = Math.max(1, Math.hypot(p1.x - p2.x, p1.y - p2.y));
      const mid = { x: (p1.x + p2.x) / 2, y: (p1.y + p2.y) / 2 };
      const v = st.ttView;
      v.s = st.ttPinch.s * (d / st.ttPinch.d);
      v.tx = mid.x - st.ttPinch.ix * v.s;
      v.ty = mid.y - st.ttPinch.iy * v.s;
      clampTTView(v);
      drawTT();
      return;
    }
    if (!st.ttDrag) return;
    const dx = e.clientX - st.ttDrag.x;
    const dy = e.clientY - st.ttDrag.y;
    if (Math.abs(dx) > 4 || Math.abs(dy) > 4) st.ttDrag.moved = true;
    // horizontal drag spins; vertical drag pans when zoomed in
    st.ttFrame = Math.round(st.ttDrag.frame - dx / 6);
    if (st.ttView.s > 1.01) {
      const c = ttRef.current;
      const rect = c.getBoundingClientRect();
      st.ttView.ty = st.ttDrag.ty + dy * (c.height / rect.height);
      clampTTView(st.ttView);
    }
    drawTT();
  }
  function onTTUp(e) {
    const st = S.current;
    if (!st) return;
    if (st.ttPan) {
      st.ttPan = null;
      return;
    }
    st.ttPts.delete(e.pointerId);
    if (st.ttPts.size < 2) st.ttPinch = null;
    if (!st.ttDrag) return;
    const wasTap = !st.ttDrag.moved;
    st.ttDrag = null;
    const tt = st.m.turntable;
    if (wasTap && tt.planes.length) {
      // pick the plane whose ring center is nearest the tap: works for any
      // ring spacing (the rings' big tilted ellipses overlap heavily)
      const [, y] = eventToImageXY(e, ttRef.current);
      const iy = (y - st.ttView.ty) / st.ttView.s;
      let best = 0;
      tt.planes.forEach((p, i) => {
        if (Math.abs(iy - p.screenY) < Math.abs(iy - tt.planes[best].screenY))
          best = i;
      });
      selectSlice(best);
    }
  }
  function onTTDblClick() {
    const st = S.current;
    if (!st) return;
    st.ttView = { s: 1, tx: 0, ty: 0 };
    drawTT();
  }

  const sliceMeta = S.current?.m.slices ?? [
    { label: "Top" },
    { label: "Middle" },
    { label: "Bottom" },
  ];
  const hasTT = !!S.current?.ttImg;
  const userDots = S.current
    ? S.current.dots.reduce((a, d) => a + d.length, 0)
    : 0;
  const autoBusy = !!S.current?.autoTimer;
  const slicesFinished = sliceMeta.map((_, i) => sliceReachedEnd(i));
  const nextUnfinished = slicesFinished.findIndex((f) => !f);
  // a finished slice announces itself ON the canvas: dot-dropping pauses
  // behind a veil with the next step one tap away — some visitors keep
  // extending a done trace because they never read the status line
  const sliceCue =
    phase === "play" &&
    opened &&
    slicesFinished[active] &&
    !cueDismissed[active] &&
    !autoBusy;

  return (
    <div className="vc-gs-demo" id="seg-demo">
      {phase === "idle" || phase === "loading" || phase === "error" ? (
        <div className="vc-gs-demo__poster">
          <img
            src={base + "slice_middle.webp"}
            alt="CT cross-section of a rolled scroll: a spiral of carbonized papyrus"
            loading="lazy"
            decoding="async"
          />
          <div className="vc-gs-demo__poster-overlay">
            <p>
              This is a CT slice straight through a rolled scroll. Every
              bright ring is the same sheet of papyrus, wound around
              itself. Trace one layer of it across three slices and we'll
              flatten it to show what's written on it.
            </p>
            <button
              className="vc-btn"
              onClick={start}
              disabled={phase === "loading"}
            >
              {phase === "loading"
                ? "Loading the volume…"
                : phase === "error"
                  ? "Retry"
                  : "Trace the sheet"}
            </button>
            {phase === "error" && (
              <p className="vc-gs-demo__err">
                Couldn't load the demo data. You can still explore the{" "}
                <a href="/data_browser">Data Browser</a>.
              </p>
            )}
          </div>
        </div>
      ) : null}

      <div
        className="vc-gs-demo__stage"
        style={{
          display:
            phase === "play" || phase === "done" ? undefined : "none",
        }}
      >
        {/* the three moves, always visible, so nobody is ever lost */}
        <ol className="vc-gs-steps" aria-label="How this demo works">
          {[
            ["Pick a plane on the scroll", opened],
            [
              "Follow one layer with dots",
              slicesFinished.every(Boolean),
            ],
            ["Read the flattened sheet", false],
          ].map(([label, done], i, arr) => {
            const on = !done && (i === 0 || arr[i - 1][1]);
            return (
              <li
                key={i}
                className={`vc-gs-step${done ? " vc-gs-step--done" : ""}${on ? " vc-gs-step--on" : ""}`}
              >
                <span className="vc-gs-step__n">{done ? "✓" : i + 1}</span>
                {label}
              </li>
            );
          })}
        </ol>
        <div className="vc-gs-seg">
          {/* the segment in 3D + plane picker */}
          <div className="vc-gs-seg__scroll" ref={scrollPaneRef}>
            {hasTT ? (
              <canvas
                ref={ttRef}
                className="vc-gs-seg__ttcanvas"
                role="img"
                aria-label="The scroll in 3D, its layers visible. Drag to rotate; tap a plane ring to open that cross-section."
                onPointerDown={onTTDown}
                onPointerMove={onTTMove}
                onPointerUp={onTTUp}
                onPointerCancel={onTTUp}
                onDoubleClick={onTTDblClick}
                onContextMenu={(e) => e.preventDefault()}
              />
            ) : (
              <svg
                viewBox="0 0 120 300"
                className="vc-gs-seg__svg"
                aria-hidden="true"
              >
                <rect
                  x="25"
                  y="30"
                  width="70"
                  height="240"
                  rx="35"
                  fill="none"
                  stroke="var(--vc-line, #333)"
                  strokeWidth="1.5"
                />
                <ellipse
                  cx="60"
                  cy="30"
                  rx="35"
                  ry="12"
                  fill="none"
                  stroke="var(--vc-line, #333)"
                  strokeWidth="1.5"
                />
                <path
                  d="M60 30 m-24 0 a24 8 0 1 1 48 0 a18 6 0 1 1 -36 0 a12 4 0 1 1 24 0 a6 2 0 1 1 -12 0"
                  fill="none"
                  stroke="var(--vc-line, #333)"
                  strokeWidth="1"
                  opacity="0.7"
                />
                {[75, 150, 225].map((y, i) => {
                  const on = active === i;
                  const done = slicesFinished[i];
                  return (
                    <g
                      key={i}
                      onClick={() => selectSlice(i)}
                      style={{ cursor: "pointer" }}
                    >
                      <ellipse
                        cx="60"
                        cy={y}
                        rx="42"
                        ry="14"
                        fill={
                          on
                            ? "rgba(229,80,43,0.14)"
                            : "rgba(255,255,255,0.03)"
                        }
                        stroke={on ? "#E5502B" : "var(--vc-line, #444)"}
                        strokeWidth={on ? "2" : "1"}
                      />
                      {done && (
                        <text
                          x="14"
                          y={y + 4}
                          textAnchor="middle"
                          fontSize="12"
                          fill="#E5502B"
                        >
                          ✓
                        </text>
                      )}
                    </g>
                  );
                })}
              </svg>
            )}
            {hasTT && (
              <p className="vc-gs-seg__ttcap vc-gs-dim">
                The scroll, its layers visible.{" "}
                {coarse
                  ? "Drag to spin, pinch to zoom (double-tap resets), tap a plane to open it."
                  : "Drag to spin, scroll to zoom (double-click resets), click a plane to open it."}
              </p>
            )}
          </div>

          {/* cross-section */}
          <figure className="vc-gs-demo__pane">
            <div className="vc-gs-slicewrap">
              <canvas
                ref={sliceRef}
                className="vc-gs-demo__canvas vc-gs-demo__canvas--draw"
                onPointerDown={onSliceDown}
                onPointerMove={onSliceMove}
                onPointerUp={onSliceUp}
                onPointerCancel={onSliceUp}
                onContextMenu={(e) => e.preventDefault()}
              />
              {sliceCue && (
                <div className="vc-gs-slicedone">
                  <span className="vc-gs-slicedone__check" aria-hidden="true">
                    ✓
                  </span>
                  <p>{sliceMeta[active].label} plane traced.</p>
                  {nextUnfinished >= 0 && (
                    <button className="vc-btn" onClick={goPickNextPlane}>
                      Trace the next layer
                    </button>
                  )}
                  <button
                    className="vc-gs-slicedone__dismiss"
                    onClick={() =>
                      setCueDismissed((c) => ({ ...c, [active]: true }))
                    }
                  >
                    Stay on this slice
                  </button>
                </div>
              )}
            </div>
            <figcaption>
              {opened
                ? `Cross-section, ${sliceMeta[active].label?.toLowerCase()} plane · tap to drop a dot · tap a dot to remove it · pinch or scroll to zoom`
                : "Cross-section. Pick a plane on the scroll to open it."}
            </figcaption>
          </figure>

          {/* flattened render */}
          <figure className="vc-gs-demo__pane">
            <canvas ref={renderRef} className="vc-gs-demo__canvas" />
            <figcaption>The flattened sheet, built from your dots.</figcaption>
          </figure>
        </div>

        <div className="vc-gs-demo__bar">
          <div className="vc-gs-demo__tools">
            {opened && (
              <button className="vc-gs-tool" onClick={clearSlice}>
                Clear this slice
              </button>
            )}
            {/* only offered once they've genuinely tried: the point of the
                demo is doing it, not watching it */}
            {phase === "play" && userDots >= 4 && !autoBusy && (
              <button className="vc-gs-tool" onClick={finishForMe}>
                Finish it for me
              </button>
            )}
          </div>
          <label className="vc-gs-hint">
            <span>Hints</span>
            <input
              type="range"
              min="0"
              max="100"
              value={hintLevel}
              aria-label="Hint level"
              onChange={(e) => {
                const v = +e.target.value;
                hintRef.current = v;
                setHintLevel(v);
                drawSlice(activeRef.current);
                drawTTRef.current(); // the plane pulse honors the slider too
              }}
            />
          </label>
          <div
            className="vc-gs-meter"
            role="progressbar"
            aria-valuenow={Math.round(
              (coverage.reduce((a, b) => a + b, 0) / 3) * 100,
            )}
            aria-valuemin={0}
            aria-valuemax={100}
            aria-label="Surface traced"
          >
            <div
              className="vc-gs-meter__fill"
              style={{
                width: `${Math.min(100, Math.round((coverage.reduce((a, b) => a + b, 0) / 3 / DONE_COVERAGE) * 100 * 0.6))}%`,
              }}
            />
            <span className="vc-gs-meter__label">
              surface traced{" "}
              {Math.round((coverage.reduce((a, b) => a + b, 0) / 3) * 100)}%
            </span>
          </div>
        </div>
        <p className="vc-gs-demo__status" aria-live="polite">
          {status}
          {phase === "play" && nextUp != null && (
            <button className="vc-btn vc-gs-nextlayer" onClick={goPickNextPlane}>
              Trace the next layer
            </button>
          )}
        </p>

        {/* phone finish pill: the payoff renders below the fold, so the
            finish announces itself over whatever the user is doing and a
            tap goes there (CSS hides this on desktop) */}
        {phase === "done" && !spiralOn && !doneCueSeen && (
          <div className="vc-gs-donecue">
            <button
              className="vc-btn"
              onClick={() => {
                setDoneCueSeen(true);
                setTimeout(
                  () =>
                    outroRef.current?.scrollIntoView({
                      behavior: "smooth",
                      block: "center",
                    }),
                  60,
                );
              }}
            >
              ✓ All three planes traced — see what you built
            </button>
          </div>
        )}

        {phase === "done" && (
          <div className="vc-gs-demo__outro" ref={outroRef}>
            <p className="vc-gs-demo__win">
              Those letters sat inside the roll, unreadable, until you traced
              their sheet. Real segments run through hundreds of turns and
              meters of papyrus; tracing them is still semi-manual.
              Automating what you just did by hand is the{" "}
              <a href="/2026_open_problems#2-unwrapping-turning-disconnected-voxels-into-a-surface">
                open problem
              </a>{" "}
              worth the{" "}
              <a href="/prizes#2027-grand-prize">$1,000,000 Grand Prize</a>.
            </p>
            {spiralOn ? (
              <UnrollReveal />
            ) : (
              <div className="vc-gs-demo__reward vc-gs-spiral__invite">
                <span>
                  Your sheet keeps going past this window. Unroll the whole
                  strip and watch the ink model read it.
                </span>
                <button className="vc-btn" onClick={() => setSpiralOn(true)}>
                  Unroll it
                </button>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
