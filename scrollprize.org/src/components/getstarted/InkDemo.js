import React, { useEffect, useRef, useState } from "react";
import useBaseUrl from "@docusaurus/useBaseUrl";
import {
  loadImage,
  loadManifest,
  imageToGray,
  makeCanvas,
  eventToImageXY,
  drawPill,
  pillLabel,
  useCoarsePointer,
} from "./demoUtils";

/*
 * "Can you find the ink?" — live ink detection on the real porphyras patch
 * (Scroll 1, PHerc. Paris 4: the first word ever read from a sealed scroll).
 *
 * The user paints on the real surface render (top pane). Every brush stamp
 * is scored against the label mask — the model's own prediction thresholded
 * at 200, which is never displayed as ground truth. The bottom pane shows
 * the model's current prediction; an "Overlay prediction" toggle onion-skins
 * the same prediction (tinted purple — porphyras) over the surface in the
 * top pane. What the prediction shows is driven by TOTAL LABELED AREA —
 * right and wrong coverage, never letter completion — and every driver is
 * continuous, no reveal cliffs:
 *   - around correct labels, broken stroke-fragments of the prediction
 *     light up (a fine fragment field breaks the reveal; it solidifies as
 *     coverage grows), spilling past the letter you're painting;
 *   - patches distributed across the WHOLE image fade in with labeled
 *     area from the first strokes (a blurred-noise order field decides
 *     which parts emerge first).
 * Labeling bare papyrus teaches the model that papyrus is ink: real
 * false-positive noise floods the WHOLE prediction and drowns the ink
 * signal. Erasing those strokes removes the bad labels and the model
 * recovers — the one place erasing changes the model, since correct
 * training (labeledMask) is monotonic like the real thing. The pane
 * behaves like a real training loop: strokes are batched, a dimmed
 * "Generating prediction…" moment runs, and the WHOLE picture jumps at
 * once. Each inference re-rolls the false positives — the weights moved,
 * so it's new noise in new places (the two orientations of a real no-ink
 * prediction are recombined under fresh flips/rotations/translations and
 * random blob masks), never the old noise sliding around.
 *
 * Data contract: static/get-started/ink/manifest.json (see
 * scripts/genGetStartedMock.py).
 */

const INK_FRAC_CORRECT = 0.35; // stamp counts as "on ink" above this
const LETTER_LEARN_FRAC = 0.4; // letter counts as taught above this coverage
const LETTERS_TO_GENERALIZE = 3;
const AREA_DONE_FRAC = 0.2; // prediction completes after 20% of ink is labeled
const WRONG_FRAC_MAX = 0.4; // …with under 40% of the labels FAR from any ink
const FAR_INK_RADIUS = 2.5; // in brush radii: bad labels beyond this are "far"
const NOISE_GAIN = 1.5; // false positives read as CONFIDENT, near-white blobs
const MAX_ZOOM = 5; // left pane: wheel/pinch zoom for finer labeling
const PRED_TINT = "#8d00ff"; // overlay color: porphyras

export default function InkDemo() {
  const base = useBaseUrl("/get-started/ink/");
  const [phase, setPhase] = useState("idle"); // idle|loading|play|done|error
  const [tool, setTool] = useState("brush"); // brush|eraser
  const [labeledPct, setLabeledPct] = useState(0);
  const [status, setStatus] = useState(
    "Paint over the bright, cracked texture. That's dried ink. Label at least 20% of it and stay off the bare papyrus.",
  );
  const [generalized, setGeneralized] = useState(false);
  const [hintOn, setHintOn] = useState(false); // a hint target is on screen
  const [readOn, setReadOn] = useState(false); // done-state reveal opened
  const [doneCueSeen, setDoneCueSeen] = useState(false); // phone finish pill tapped
  const [overlay, setOverlay] = useState(true); // prediction over the surface
  const coarse = useCoarsePointer(); // gesture wording matches the device

  const drawRef = useRef(null); // visible surface canvas (top pane)
  const predRef = useRef(null); // prediction canvas (bottom pane)
  const statusRef = useRef(null); // status line, scrolled to on phones
  const outroRef = useRef(null); // done payoff, scrolled to on phones
  const S = useRef(null); // demo state (never re-rendered)
  const overlayRef = useRef(overlay);
  overlayRef.current = overlay;
  const toolRef = useRef(tool);
  toolRef.current = tool;
  const [autoBusy, setAutoBusy] = useState(false);

  async function start() {
    setPhase("loading");
    try {
      const m = await loadManifest(base);
      const [papyrus, prediction, label, noiseImg] = await Promise.all([
        loadImage(base + m.papyrus),
        loadImage(base + m.prediction),
        loadImage(base + m.label),
        loadImage(base + m.noise),
      ]);
      const { width: W, height: H } = m;
      // on phones the draw pane becomes a TALL viewport onto the strip
      // (roughly 40% of the screen), opened zoomed-in — painting on an
      // 80px-tall full-strip view is hopeless
      const narrow = window.innerWidth < 700;
      const dcW = W;
      const dcH = narrow
        ? Math.round(
            (W * window.innerHeight * 0.4) /
              Math.max(280, window.innerWidth - 34),
          )
        : H;
      const minS = Math.max(1, dcH / H);
      const labelData = imageToGray(label, W, H);
      let inkTotal = 0;
      for (let i = 0; i < labelData.length; i++)
        if (labelData[i] > 127) inkTotal++;

      S.current = {
        m,
        W,
        H,
        papyrus,
        prediction,
        label,
        labelData,
        inkTotal,
        brush: Math.max(6, Math.round(m.brushRadius / 2)), // drawn stroke radius
        labeledMask: new Uint8Array(W * H), // unique ink px the user labeled
        labeled: 0,
        strokes: makeCanvas(W, H), // ember stroke overlay
        reveal: makeCanvas(W, H), // strictly-local reveal around correct strokes
        localRev: makeCanvas(W, H), // reveal broken by the fragment field
        fragMask: makeFragMask(W, H, 0x1f83d9ab, 0x85ebca6b), // breaks early reveals
        fragErode: makeFragMask(W, H, 0xc2b2ae35, 0x27d4eb2f), // corruption bites
        noiseC: makeCanvas(W, H), // composed noise layer (cut over signal)
        patches: makeCanvas(W, H), // generalization patches (grow with area)
        patchImg: null,
        patchGA: -1,
        revealU: makeCanvas(W, H), // union of the reveal layers
        predLayer: makeCanvas(W, H), // masked prediction, for the overlay
        dcW,
        dcH,
        minS, // baseline zoom: 1 on desktop, "strip fills the pane" on phones
        view: { s: minS, tx: (dcW - W * minS) / 2, ty: 0 }, // zoom/pan
        pointers: new Map(),
        pinch0: null,
        pendingPaint: null, // first touch buffered until it's clearly a stroke
        strokeStart: null, // current touch stroke's undo record
        stampCount: 0, // painted stamps (test hook via dataset.stamps)
        hintPt: null, // pulsing "this is ink, paint it" target
        hintedOnce: false, // first hint targets the first letter
        tmp: makeCanvas(W, H),
        orderField: makeOrderField(W, H),
        model: { A: 0, corr: 0, fpAmt: 0, wrongFrac: 0 }, // committed at each inference
        biteW: 0.5, // per-inference weighting of the two bite masks
        noiseVariant: null, // per-inference recombination of the noise
        noiseVariantP: null, // the same recombination, purple
        nvTmp: null,
        infTimer: 0,
        infDirty: false,
        generating: false, // the fake "Generating prediction…" moment
        bonusGA: 0,
        wrongStamps: [], // bad labels on bare papyrus (erasable)
        goodStamps: [], // correct labels; the reveal is rebuilt from these
        revealDirty: false, // stamps were erased — rebuild reveal at retrain
        perLetter: new Array((m.letters || []).length).fill(0),
        letterInk: null,
        generalizedDone: false,
        lastTaught: 0,
        needMoreLettersSaid: false,
        corrHealedSaid: false,
        lastPct: -1,
        painting: false,
        raf: 0,
        autoTimer: 0,
      };
      const st = S.current;
      st.patchImg = st.patches.getContext("2d").createImageData(W, H);
      // per-letter ink totals (for coverage fractions)
      st.letterInk = (m.letters || []).map((b) => {
        let n = 0;
        for (let y = b.y; y < b.y + b.h; y++)
          for (let x = b.x; x < b.x + b.w; x++)
            if (labelData[y * W + x] > 127) n++;
        return Math.max(1, n);
      });

      // the prediction with its dark background floor stripped: masked
      // reveals ADD pure ink signal over the pane's noise floor instead of
      // stacking a second gray floor (which reads as a disc around every
      // reveal window)
      const sigC = makeCanvas(W, H);
      const sgctx = sigC.getContext("2d");
      sgctx.drawImage(prediction, 0, 0);
      const sd = sgctx.getImageData(0, 0, W, H);
      const FLOOR = 40;
      const fk = 255 / (255 - FLOOR);
      for (let i = 0; i < sd.data.length; i += 4) {
        sd.data[i] = Math.max(0, (sd.data[i] - FLOOR) * fk);
        sd.data[i + 1] = Math.max(0, (sd.data[i + 1] - FLOOR) * fk);
        sd.data[i + 2] = Math.max(0, (sd.data[i + 2] - FLOOR) * fk);
      }
      sgctx.putImageData(sd, 0, 0);
      st.predSignal = sigC;

      // the same signal tinted purple, for the on-papyrus overlay
      const pp = makeCanvas(W, H);
      const ppctx = pp.getContext("2d");
      ppctx.drawImage(sigC, 0, 0);
      ppctx.globalCompositeOperation = "multiply";
      ppctx.fillStyle = PRED_TINT;
      ppctx.fillRect(0, 0, W, H);
      ppctx.globalCompositeOperation = "source-over";
      st.predPurple = pp;
      // false positives / uncertainty: a REAL model prediction over a
      // non-ink region of the same scroll (white for the prediction pane,
      // purple for the overlay). Being real output on inkless papyrus, it
      // has the exact texture and black floor of the target prediction and
      // can never spell the letters.
      const fp = makeCanvas(W, H);
      const fctx = fp.getContext("2d");
      fctx.drawImage(noiseImg, 0, 0, W, H);
      // floor-strip AND gain the raw no-ink prediction: false positives
      // must read as confident near-white blobs, not a grey wash — a
      // fooled model is sure about its hallucinations. Without the strip
      // the pane also picks up a full-frame grey cast at high noise alpha.
      {
        const fd = fctx.getImageData(0, 0, W, H);
        for (let i = 0; i < fd.data.length; i += 4) {
          fd.data[i] = Math.min(
            255,
            Math.max(0, (fd.data[i] - FLOOR) * fk * NOISE_GAIN),
          );
          fd.data[i + 1] = Math.min(
            255,
            Math.max(0, (fd.data[i + 1] - FLOOR) * fk * NOISE_GAIN),
          );
          fd.data[i + 2] = Math.min(
            255,
            Math.max(0, (fd.data[i + 2] - FLOOR) * fk * NOISE_GAIN),
          );
        }
        fctx.putImageData(fd, 0, 0);
      }
      st.fp = fp;
      // purple copy for the on-papyrus overlay — fp is already stripped
      // and gained, so only the tint is left to apply
      const fpP = makeCanvas(W, H);
      const fpctx = fpP.getContext("2d");
      fpctx.drawImage(fp, 0, 0);
      fpctx.globalCompositeOperation = "multiply";
      fpctx.fillStyle = PRED_TINT;
      fpctx.fillRect(0, 0, W, H);
      fpctx.globalCompositeOperation = "source-over";
      st.fpP = fpP;
      // the same noise flipped both ways: every inference recombines the
      // two orientations under fresh transforms and blob masks, so no two
      // predictions show the same false positives
      const flip = (src) => {
        const c = makeCanvas(W, H);
        const cctx = c.getContext("2d");
        cctx.translate(W, H);
        cctx.scale(-1, -1);
        cctx.drawImage(src, 0, 0);
        return c;
      };
      st.fp2 = flip(fp);
      st.fpP2 = flip(fpP);
      st.fragErode2 = flip(st.fragErode);

      const dc = drawRef.current;
      const pc = predRef.current;
      dc.width = dcW;
      dc.height = dcH;
      pc.width = W;
      pc.height = H;
      redrawLeft();
      setPhase("play");
      composite();
    } catch (e) {
      console.error(e);
      setPhase("error");
    }
  }

  function stamp(x, y) {
    const st = S.current;
    if (!st) return;
    const { m, W, H, labelData } = st;
    const r = st.brush;
    const sctx = st.strokes.getContext("2d");

    if (toolRef.current === "eraser") {
      // erases strokes and takes labels OUT of the training set — bad ones
      // (the model heals) and good ones alike. The prediction is NOT edited
      // here: the reveal is rebuilt from the surviving stamps at the next
      // retrain, so erasing a label makes the model lose everything it saw
      // AROUND that label, not just the pixels under the eraser.
      {
        const ctx = st.strokes.getContext("2d");
        ctx.globalCompositeOperation = "destination-out";
        ctx.beginPath();
        ctx.arc(x, y, r * 2, 0, Math.PI * 2);
        ctx.fillStyle = "rgba(0,0,0,1)";
        ctx.fill();
        ctx.globalCompositeOperation = "source-over";
      }
      const er = r * 2;
      const ex0 = Math.max(0, Math.round(x - er));
      const ex1 = Math.min(W - 1, Math.round(x + er));
      const ey0 = Math.max(0, Math.round(y - er));
      const ey1 = Math.min(H - 1, Math.round(y + er));
      for (let yy = ey0; yy <= ey1; yy++)
        for (let xx = ex0; xx <= ex1; xx++) {
          const dx = xx - x;
          const dy = yy - y;
          if (dx * dx + dy * dy > er * er) continue;
          const p = yy * W + xx;
          if (st.labeledMask[p]) {
            st.labeledMask[p] = 0;
            st.labeled--;
            (st.m.letters || []).forEach((b, i) => {
              if (xx >= b.x && xx < b.x + b.w && yy >= b.y && yy < b.y + b.h)
                st.perLetter[i] = Math.max(0, st.perLetter[i] - 1);
            });
          }
        }
      const er2 = (r * 2.2) ** 2;
      const goodBefore = st.goodStamps.length;
      st.goodStamps = st.goodStamps.filter(
        (p) => (p.x - x) ** 2 + (p.y - y) ** 2 > er2,
      );
      if (st.goodStamps.length !== goodBefore) st.revealDirty = true;
      const before = st.wrongStamps.length;
      st.wrongStamps = st.wrongStamps.filter(
        (p) => (p.x - x) ** 2 + (p.y - y) ** 2 > er2,
      );
      if (before && !st.wrongStamps.length && !st.corrHealedSaid) {
        st.corrHealedSaid = true;
        st.wrongWarnOn = false;
        setStatus("Bad labels gone. The false positives clear up.");
      }
      const pct = Math.round((st.labeled / st.inkTotal) * 100);
      if (pct !== st.lastPct) {
        st.lastPct = pct;
        setLabeledPct(pct);
      }
      redrawLeft();
      scheduleInference(); // removing training data retrains the model too
      return;
    }

    // paint the visible stroke
    st.stampCount++;
    sctx.beginPath();
    sctx.arc(x, y, r, 0, Math.PI * 2);
    sctx.fillStyle = "rgba(229,80,43,0.45)";
    sctx.fill();

    // score the stamp against the label
    const x0 = Math.max(0, Math.round(x - r));
    const x1 = Math.min(W - 1, Math.round(x + r));
    const y0 = Math.max(0, Math.round(y - r));
    const y1 = Math.min(H - 1, Math.round(y + r));
    let ink = 0;
    let tot = 0;
    for (let yy = y0; yy <= y1; yy++)
      for (let xx = x0; xx <= x1; xx++) {
        const dx = xx - x;
        const dy = yy - y;
        if (dx * dx + dy * dy > r * r) continue;
        tot++;
        if (labelData[yy * W + xx] > 127) ink++;
      }
    if (!tot) return;

    const frac = ink / tot;
    if (frac > INK_FRAC_CORRECT) {
      // count only NEW ink pixels — going over the same spot twice
      // doesn't grow the training set
      let fresh = 0;
      for (let yy = y0; yy <= y1; yy++)
        for (let xx = x0; xx <= x1; xx++) {
          const dx = xx - x;
          const dy = yy - y;
          if (dx * dx + dy * dy > r * r) continue;
          const p = yy * W + xx;
          if (labelData[p] > 127 && !st.labeledMask[p]) {
            st.labeledMask[p] = 1;
            fresh++;
            // touch strokes stay undoable until the pointer lifts: a late
            // second finger means this was a pan attempt, not labeling
            if (st.strokeStart) st.strokeStart.fresh.push(p);
          }
        }
      st.labeled += fresh;
      // back on real ink: the bare-papyrus warning has done its job — it
      // shouldn't nag for the rest of the session (letter/teaching messages
      // below may still replace the cleared status)
      if (st.wrongWarnOn) {
        st.wrongWarnOn = false;
        setStatus("");
      }
      // any painting dismisses the hint — the user is off exploring, so
      // the ring and its message shouldn't keep nagging; hitting the
      // target itself earns the confirmation line
      if (st.hintPt) {
        const found =
          Math.hypot(x - st.hintPt.x, y - st.hintPt.y) < r * 2.5;
        clearHint(found);
        if (!found && !st.wrongWarnOn) setStatus("");
      }
      // local reveal: the prediction lights up AROUND where you labeled,
      // not just under the brush — a wide soft halo that spills past the
      // letter you're painting into its neighbors. The fragment field
      // breaks it into stroke-fragments at composite time, so it never
      // reads as a disc or as your stroke echoed back. Stamps are thinned
      // and kept in goodStamps, and the halo is drawn once per KEPT stamp:
      // the reveal stays a pure function of the stamp list, so erasing
      // stamps later rebuilds it exactly (minus their whole halos).
      const lastG = st.goodStamps[st.goodStamps.length - 1];
      if (!lastG || (lastG.x - x) ** 2 + (lastG.y - y) ** 2 > (r * 0.6) ** 2) {
        st.goodStamps.push({ x, y });
        paintHalo(st.reveal.getContext("2d"), x, y, r);
      }
      // letter coverage (status/teaching only — never drives the reveal)
      if (fresh) {
        (m.letters || []).forEach((b, i) => {
          if (x >= b.x && x < b.x + b.w && y >= b.y && y < b.y + b.h)
            st.perLetter[i] += fresh;
        });
        maybeGeneralize();
      }
    } else {
      // a bad label: the model starts learning that papyrus is ink, and
      // false positives light up across the WHOLE prediction. Thin the
      // stamps (drags fire dozens) so erasing can take them back out.
      // Slop hugging a letter's edge is normal labeling, not sabotage:
      // stamps are tagged by whether any ink sits within reach, and only
      // the FAR kind can block the finish (commitModel).
      const last = st.wrongStamps[st.wrongStamps.length - 1];
      if (!last || (last.x - x) ** 2 + (last.y - y) ** 2 > r * r)
        st.wrongStamps.push({
          x,
          y,
          far: !inkNear(labelData, W, H, x, y, r * FAR_INK_RADIUS),
        });
      st.corrHealedSaid = false;
      st.wrongWarnOn = true;
      clearHint(false); // stop the ring pulsing under the warning too
      setStatus(
        "That's bare papyrus. The model now thinks papyrus looks like ink, so false positives light up everywhere. Erase those strokes and it recovers.",
      );
    }
    // the label meter tracks YOUR labels live; the prediction pane only
    // refreshes on the inference clock
    const pct = Math.round((st.labeled / st.inkTotal) * 100);
    if (pct !== st.lastPct) {
      st.lastPct = pct;
      setLabeledPct(pct);
    }
    redrawLeft();
    scheduleInference();
  }

  function maybeGeneralize() {
    const st = S.current;
    const { m } = st;
    if (!(m.letters || []).length) return;
    const taught = st.perLetter.filter(
      (v, i) => v / st.letterInk[i] > LETTER_LEARN_FRAC,
    ).length;
    if (taught > st.lastTaught && taught < LETTERS_TO_GENERALIZE) {
      setStatus(
        `Letter learned (${taught}/${LETTERS_TO_GENERALIZE}). The model only trusts regions like the ones you've shown it. Teach it another.`,
      );
    }
    st.lastTaught = Math.max(st.lastTaught, taught);
    if (!st.generalizedDone && taught >= LETTERS_TO_GENERALIZE) {
      // purely informational: the reveal drivers are continuous, so this
      // moment changes the message, not the picture
      st.generalizedDone = true;
      setGeneralized(true);
      setStatus(
        "The model generalized. Regions you never labeled are appearing; more labels, more detail.",
      );
    }
  }

  function animateBonus(to, ms) {
    const st = S.current;
    const from = st.bonusGA;
    const t0 = performance.now();
    const step = () => {
      if (!S.current) return;
      const k = Math.min(1, (performance.now() - t0) / ms);
      st.bonusGA = from + (to - from) * k * (2 - k);
      requestComposite();
      if (k < 1) requestAnimationFrame(step);
    };
    requestAnimationFrame(step);
  }

  function redrawLeft() {
    const st = S.current;
    const c = drawRef.current;
    const ctx = c.getContext("2d");
    const v = st.view;
    // canvas-backing px per CSS px, so rings/labels keep a constant
    // on-screen size at any zoom on any device
    const k = c.width / (c.getBoundingClientRect().width || c.width);
    // testable draw state (mirrors SegDemo's ttview/drawnPlane hooks)
    c.dataset.view = `${v.s.toFixed(3)},${v.tx.toFixed(1)},${v.ty.toFixed(1)}`;
    c.dataset.stamps = String(st.stampCount);
    c.dataset.marks = `${st.goodStamps.length},${st.wrongStamps.length},${
      st.wrongStamps.filter((p) => p.far).length
    }`;
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.fillStyle = "#000";
    ctx.fillRect(0, 0, c.width, c.height);
    ctx.setTransform(v.s, 0, 0, v.s, v.tx, v.ty);
    ctx.drawImage(st.papyrus, 0, 0);
    if (overlayRef.current) {
      // light the predicted ink up over the surface; screen blend keeps
      // the model's dark background from muddying the papyrus. Two passes:
      // the second saturates the purple so it reads over busy texture.
      ctx.globalCompositeOperation = "screen";
      ctx.drawImage(st.predLayer, 0, 0);
      ctx.globalAlpha = 0.5;
      ctx.drawImage(st.predLayer, 0, 0);
      ctx.globalAlpha = 1;
      ctx.globalCompositeOperation = "source-over";
    }
    ctx.drawImage(st.strokes, 0, 0);
    // pulsing hint target: "the ink is right here, paint it"
    if (st.hintPt && phaseRef.current === "play") {
      const ph = performance.now() / 550;
      const rr = st.brush * 2.4 + 7 * Math.sin(ph);
      ctx.beginPath();
      ctx.arc(st.hintPt.x, st.hintPt.y, rr, 0, Math.PI * 2);
      ctx.strokeStyle = `rgba(255,106,61,${0.9 - 0.25 * Math.sin(ph)})`;
      ctx.lineWidth = (2.5 * k) / v.s;
      ctx.stroke();
      ctx.beginPath();
      ctx.arc(st.hintPt.x, st.hintPt.y, rr + 9, 0, Math.PI * 2);
      ctx.strokeStyle = "rgba(255,106,61,0.3)";
      ctx.lineWidth = (1.2 * k) / v.s;
      ctx.stroke();
      // first-time affordance: a labeled pill next to the ring, gone
      // after the first stamp — some visitors won't read the copy
      if (st.labeled === 0 && st.wrongStamps.length === 0)
        drawPill(ctx, pillLabel(), st.hintPt.x, st.hintPt.y, rr, k / v.s, st.W);
    }
    ctx.setTransform(1, 0, 0, 1, 0, 0);
  }

  // ---- left-pane zoom (wheel at cursor, pinch on touch) -------------------

  function clampView(v) {
    const st = S.current;
    // phones OPEN zoomed to fill the tall pane (minS), but that's a
    // convenience, not a floor: zooming back out to the whole strip
    // (s = 1, the desktop framing) stays available, with the strip
    // centered on whichever axis it no longer fills
    v.s = Math.min(st.minS * MAX_ZOOM, Math.max(1, v.s));
    v.tx =
      st.W * v.s <= st.dcW
        ? (st.dcW - st.W * v.s) / 2
        : Math.min(0, Math.max(st.dcW - st.W * v.s, v.tx));
    v.ty =
      st.H * v.s <= st.dcH
        ? (st.dcH - st.H * v.s) / 2
        : Math.min(0, Math.max(st.dcH - st.H * v.s, v.ty));
  }

  function pointerToImg(e) {
    const [cx, cy] = eventToImageXY(e, drawRef.current);
    const v = S.current.view;
    return [(cx - v.tx) / v.s, (cy - v.ty) / v.s];
  }

  // non-passive wheel listener (React's onWheel can't preventDefault)
  useEffect(() => {
    const c = drawRef.current;
    if (!c) return;
    const onWheel = (e) => {
      const st = S.current;
      if (!st) return;
      e.preventDefault();
      const v = st.view;
      const rect = c.getBoundingClientRect();
      const cx = ((e.clientX - rect.left) / rect.width) * c.width;
      const cy = ((e.clientY - rect.top) / rect.height) * c.height;
      const ix = (cx - v.tx) / v.s;
      const iy = (cy - v.ty) / v.s;
      // cap per-event delta: free-spinning wheels send |deltaY| in the
      // thousands, and one reverse-momentum tick would slam the zoom
      v.s = v.s * Math.pow(1.0018, -Math.max(-400, Math.min(400, e.deltaY)));
      v.tx = cx - ix * v.s;
      v.ty = cy - iy * v.s;
      clampView(v);
      redrawLeft();
    };
    c.addEventListener("wheel", onWheel, { passive: false });
    return () => c.removeEventListener("wheel", onWheel);
  }, []);

  function requestComposite() {
    const st = S.current;
    if (st.raf) return;
    st.raf = requestAnimationFrame(() => {
      st.raf = 0;
      composite();
    });
  }

  function rebuildPatches(st, eGA) {
    if (Math.abs(eGA - st.patchGA) < 0.01) return;
    st.patchGA = eGA;
    const { W, H } = st;
    const d = st.patchImg.data;
    if (eGA <= 0) {
      d.fill(0);
    } else {
      const of = st.orderField;
      // wide per-pixel ramp: regions ease in over a long stretch of eGA,
      // so the reveal reads as gradual everywhere
      for (let i = 0; i < W * H; i++) {
        const a = Math.min(1, Math.max(0, (eGA * 1.28 - of[i]) / 0.3));
        const j = i * 4;
        d[j] = d[j + 1] = d[j + 2] = 255;
        d[j + 3] = a * 255;
      }
    }
    st.patches.getContext("2d").putImageData(st.patchImg, 0, 0);
  }

  function composite() {
    const st = S.current;
    if (!st || !predRef.current) return;
    const { W, H } = st;
    // everything model-driven reads the state COMMITTED at the last
    // inference (commitModel): between retrains the prediction pane holds
    // still, then the whole picture jumps at once — a training run, not a
    // paint program
    const { A, corr, fpAmt, wrongFrac } = st.model;
    // bad labels dilute the training signal a little (sig), but their real
    // damage is the erosion below: fragments of the revealed letters get
    // BITTEN OUT and noise pours into the holes
    const sig = Math.max(0.35, 1 - 0.75 * corr);
    const q = Math.min(1, A / AREA_DONE_FRAC) * (1 - 0.8 * corr);
    // a training set that's substantially bare papyrus makes the model
    // predict ink almost EVERYWHERE: the false-positive flood scales with
    // the share of bad labels, not just their count. corr saturates too
    // slowly for this (good labels dampen it) — at 50/50 the pane must
    // drown in white noise even though the letters are still under it.
    const flood = Math.min(1, wrongFrac * 1.5);

    // patches distributed over the whole image grow with TOTAL labeled
    // area from the very first strokes — no letter-completion gate, no
    // cliffs. The whole segment improves a little with every label;
    // bonusGA fills the last stretch on completion.
    const localProgress = Math.min(1, A / AREA_DONE_FRAC);
    // first-stroke bootstrap: a touch more of the prediction completes
    // around the very first labels, so the stroke you painted reads back
    // as a whole fragment instead of shreds. Fades out by ~6% coverage —
    // the pace beyond the opening is unchanged.
    const boot = 0.22 * Math.max(0, 1 - A / (AREA_DONE_FRAC * 0.3));
    const eGA = Math.min(1, Math.pow(localProgress, 1.35) * 0.9 + st.bonusGA);
    rebuildPatches(st, eGA);

    // local evidence, broken by the fragment field: early reveals are
    // scattered stroke-fragments of the prediction around your labels,
    // filling toward solid as coverage grows
    const lctx = st.localRev.getContext("2d");
    lctx.clearRect(0, 0, W, H);
    lctx.drawImage(st.reveal, 0, 0);
    lctx.globalCompositeOperation = "destination-in";
    lctx.drawImage(st.fragMask, 0, 0);
    lctx.globalCompositeOperation = "source-over";
    lctx.globalAlpha = Math.min(1, Math.pow(localProgress, 1.2) + boot);
    lctx.drawImage(st.reveal, 0, 0);
    lctx.globalAlpha = 1;

    // Union the user's local evidence with the distributed global reveal.
    const uctx = st.revealU.getContext("2d");
    uctx.clearRect(0, 0, W, H);
    // Correct signal appears promptly around the brush; only the distributed
    // prediction remains deliberately slow. With no correct label, this is
    // still zero and the pane contains noise only.
    uctx.globalAlpha =
      A > 0
        ? Math.min(1, 0.6 + 0.4 * Math.pow(localProgress, 0.8) + 0.5 * boot)
        : 0;
    uctx.drawImage(st.localRev, 0, 0);
    uctx.globalAlpha = 1;
    uctx.drawImage(st.patches, 0, 0);

    // corruption doesn't just dim the signal: it bites fragments out of
    // the reveal, and the false-positive noise pours into the holes — the
    // letters dissolve INTO noise instead of fading behind it. The bites
    // move to new spots as training steps land.
    const erode = Math.min(0.95, corr * 1.6);
    if (erode > 0.02) {
      // biteW re-rolls at every inference, so the bites land in new spots
      // with each retrain instead of animating between two poses
      for (const layer of [st.revealU, st.localRev]) {
        const ectx = layer.getContext("2d");
        ectx.globalCompositeOperation = "destination-out";
        ectx.globalAlpha = erode * st.biteW;
        ectx.drawImage(st.fragErode, 0, 0);
        ectx.globalAlpha = erode * (1 - st.biteW);
        ectx.drawImage(st.fragErode2, 0, 0);
        ectx.globalAlpha = 1;
        ectx.globalCompositeOperation = "source-over";
      }
    }

    const ctx = predRef.current.getContext("2d");
    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = "#000";
    ctx.fillRect(0, 0, W, H);

    // uncertainty floor + false positives: a real model prediction over
    // inkless papyrus, recombined FRESH at every inference (see
    // rebuildNoiseVariant) — new weights, new noise, never the old noise
    // nudged around. Where the model has real signal the noise is cut
    // back in proportion to quality, so you slowly become able to tell
    // signal from noise; corruption weakens the cut and wins the letters
    // back for the noise. The pane is pure black only before the first
    // inference.
    const nA = Math.min(0.95, 0.16 * (1 - q) + 0.05 + fpAmt + 0.8 * flood);
    // where the model is unsure, the prediction goes FAINT: the noise field
    // punches soft holes into the signal instead of sitting on top of it as
    // grey. Confidence (q) shrinks the holes; corruption reopens them.
    const holeA =
      Math.min(0.85, 0.15 + 1.4 * nA * (1 - 0.5 * q)) * (1 - 0.5 * boot);
    if (st.noiseVariant && (A > 0 || fpAmt > 0) && nA > 0.01) {
      const nctx = st.noiseC.getContext("2d");
      nctx.clearRect(0, 0, W, H);
      nctx.drawImage(st.noiseVariant, 0, 0);
      // strong cut over revealed regions: what remains of the grey lives in
      // the faint gaps of the prediction, not painted over the letters —
      // unless the training set is flooded with bad labels, in which case
      // the noise wins the letters back and sits right on top of them
      const cut = (0.6 + 0.4 * Math.pow(q, 0.55)) * sig * (1 - 0.85 * flood);
      if (cut > 0.02) {
        nctx.globalCompositeOperation = "destination-out";
        nctx.globalAlpha = cut;
        nctx.drawImage(st.revealU, 0, 0);
        nctx.globalAlpha = 1;
        nctx.globalCompositeOperation = "source-over";
      }
      ctx.globalAlpha = nA;
      ctx.drawImage(st.noiseC, 0, 0);
      // past mild flooding a screen re-pass stacks the noise on itself,
      // pushing it toward solid white as the bad-label share grows
      if (flood > 0.25) {
        ctx.globalCompositeOperation = "screen";
        ctx.globalAlpha = Math.min(1, (flood - 0.25) * 1.2);
        ctx.drawImage(st.noiseC, 0, 0);
        ctx.globalCompositeOperation = "source-over";
      }
      ctx.globalAlpha = 1;
    }

    // real prediction, masked by the reveal union and scaled by signal
    // survival; screen-blended so it ADDS to the noise floor instead of
    // replacing it — the reveal window has no visible disc edge, because
    // where the prediction is near-black the pane simply keeps its noise
    const predA =
      (0.45 + 0.55 * Math.pow(Math.min(1, A / AREA_DONE_FRAC), 0.9)) * sig +
      0.3 * boot;
    // faint patches: dig the noise field out of the prediction so the
    // uncertain spots read as MISSING signal rather than grey on top
    const punch = (c2) => {
      if (!st.noiseVariant || holeA < 0.02) return;
      c2.globalCompositeOperation = "destination-out";
      c2.globalAlpha = holeA;
      c2.drawImage(st.noiseVariant, 0, 0);
      c2.globalAlpha = 1;
      c2.globalCompositeOperation = "source-over";
    };
    const tctx = st.tmp.getContext("2d");
    tctx.clearRect(0, 0, W, H);
    tctx.globalAlpha = predA;
    tctx.drawImage(st.predSignal, 0, 0);
    tctx.globalAlpha = 1;
    tctx.globalCompositeOperation = "destination-in";
    tctx.drawImage(st.revealU, 0, 0);
    tctx.globalCompositeOperation = "source-over";
    punch(tctx);
    ctx.globalCompositeOperation = "screen";
    ctx.drawImage(st.tmp, 0, 0);
    // Immediate local feedback: a correct stroke should reveal signal where
    // the visitor painted even while global generalization is still weak.
    tctx.clearRect(0, 0, W, H);
    tctx.globalAlpha = 0.8 * sig;
    tctx.drawImage(st.predSignal, 0, 0);
    tctx.globalAlpha = 1;
    tctx.globalCompositeOperation = "destination-in";
    tctx.drawImage(st.localRev, 0, 0);
    tctx.globalCompositeOperation = "source-over";
    punch(tctx);
    ctx.drawImage(st.tmp, 0, 0);
    ctx.globalCompositeOperation = "source-over";

    // the overlay copy, in purple (masked prediction + false positives)
    const plctx = st.predLayer.getContext("2d");
    plctx.clearRect(0, 0, W, H);
    if (st.noiseVariantP && (A > 0 || fpAmt > 0) && nA > 0.01) {
      plctx.globalAlpha = nA;
      plctx.drawImage(st.noiseVariantP, 0, 0);
      if (flood > 0.25) {
        plctx.globalCompositeOperation = "screen";
        plctx.globalAlpha = Math.min(1, (flood - 0.25) * 1.2);
        plctx.drawImage(st.noiseVariantP, 0, 0);
        plctx.globalCompositeOperation = "source-over";
      }
      plctx.globalAlpha = 1;
    }
    tctx.clearRect(0, 0, W, H);
    tctx.globalAlpha = predA;
    tctx.drawImage(st.predPurple, 0, 0);
    tctx.globalAlpha = 1;
    tctx.globalCompositeOperation = "destination-in";
    tctx.drawImage(st.revealU, 0, 0);
    tctx.globalCompositeOperation = "source-over";
    punch(tctx);
    plctx.globalCompositeOperation = "screen";
    plctx.drawImage(st.tmp, 0, 0);
    plctx.globalCompositeOperation = "source-over";

    if (overlayRef.current) redrawLeft();

    // labeled area alone completes the demo — the letter milestones are
    // narrative beats along the way, never a gate (a user who labels the
    // target share of the ink has taught it the letters, wherever the
    // strokes landed). Compared on the ROUNDED percent so done fires
    // exactly when the meter reads the threshold. The goal is two-sided:
    // AREA_DONE_FRAC of the ink labeled, with under WRONG_FRAC_MAX of the
    // training set far out on bare papyrus (edge slop never counts).
    const pctReached =
      Math.round(A * 100) >= Math.round(AREA_DONE_FRAC * 100);
    const clean = wrongFrac < WRONG_FRAC_MAX;
    if (!pctReached || clean) st.dirtyDoneSaid = false; // message can re-arm
    if (pctReached && clean && phaseRef.current === "play") {
      setPhase("done");
      setStatus("");
      animateBonus(1, 1600);
    } else if (pctReached && !clean && phaseRef.current === "play") {
      if (!st.dirtyDoneSaid) {
        st.dirtyDoneSaid = true;
        setStatus(
          "Enough ink is labeled, but over 40% of your labels sit out on bare papyrus, far from any ink. They're poisoning the model — erase them to finish.",
        );
        // on phones the status line sits below the fold — a blocking
        // message the user can't see is a dead end, so bring it in
        if (window.innerWidth < 700)
          setTimeout(
            () =>
              statusRef.current?.scrollIntoView({
                behavior: "smooth",
                block: "nearest",
              }),
            600,
          );
      }
    } else if (!pctReached && phaseRef.current === "done") {
      // the model is a function of the current training set: erase your
      // labels back below the bar and the finished state unwinds with them
      setPhase("play");
      setReadOn(false);
      setDoneCueSeen(false); // re-finishing should announce itself again
      animateBonus(0, 600);
    }
    if (st.generating) drawGenerating();
  }

  // ---- the inference clock -------------------------------------------------
  // Strokes don't repaint the prediction; they mark the model dirty. A beat
  // later the pane freezes under "Generating prediction…", then the WHOLE
  // output jumps to the new state at once — exactly how retrain-and-infer
  // feels, and it licenses the noise to be completely different each time.

  function scheduleInference() {
    const st = S.current;
    if (!st) return;
    st.infDirty = true;
    if (st.infTimer || st.generating) return;
    st.infTimer = setTimeout(runInference, 550);
  }

  function runInference() {
    const st = S.current;
    if (!st) return;
    st.infTimer = 0;
    st.infDirty = false;
    st.generating = true;
    drawGenerating();
    setTimeout(() => {
      const cur = S.current;
      if (!cur) return;
      cur.generating = false;
      commitModel(cur);
      composite();
      // strokes that landed while this inference ran get the next one
      if (cur.infDirty) {
        cur.infDirty = false;
        cur.infTimer = setTimeout(runInference, 420);
      }
    }, 430);
  }

  function commitModel(st) {
    // erased stamps leave the training set here: the reveal is rebuilt
    // from what survives, so their whole halos vanish in one retrain —
    // the model forgets what it saw around the erased labels, it doesn't
    // get a hole poked where the eraser touched
    if (st.revealDirty) {
      st.revealDirty = false;
      const rctx = st.reveal.getContext("2d");
      rctx.clearRect(0, 0, st.W, st.H);
      for (const p of st.goodStamps) paintHalo(rctx, p.x, p.y, st.brush);
    }
    const A = st.labeled / st.inkTotal; // grounded: labeled ink area
    // bad labels corrupt the model in proportion to their share of the
    // training set: one stray stroke on bare papyrus visibly pollutes the
    // next prediction, piling on good labels drowns it out, and erasing
    // the stroke heals it (exactly the real failure mode)
    const wrongArea = st.wrongStamps.length * Math.PI * st.brush * st.brush;
    const corr = wrongArea / (wrongArea + st.labeled * 1.5 + 900);
    // the done gate (and the flood) count only labels FAR from any ink:
    // sloppy strokes along a letter's edge still pollute the picture a
    // little (corr, above, sees every wrong stamp), but they can't block
    // the finish — hitting 40% far labels takes deliberate scribbling
    // out on the open papyrus
    const farArea =
      st.wrongStamps.filter((p) => p.far).length *
      Math.PI *
      st.brush *
      st.brush;
    const wrongFrac = farArea / Math.max(1, farArea + st.labeled);
    // sparse-data over-prediction: light scattered noise from the first
    // strokes that thins out as the real labeled area grows
    const sparse =
      Math.min(1, st.labeled / 900) *
      Math.max(0, 1 - A / (AREA_DONE_FRAC * 0.7));
    const fpAmt = Math.min(0.9, Math.max(0.95 * corr, 0.5 * sparse));
    st.model = { A, corr, fpAmt, wrongFrac };
    // committed gate state, testable (mirrors the dataset.view hooks)
    if (drawRef.current)
      drawRef.current.dataset.wrongfrac = wrongFrac.toFixed(3);
    st.biteW = Math.random();
    rebuildNoiseVariant(st);
  }

  // one inference's worth of false positives: the two noise orientations
  // under fresh random flips/rotations/translations, each shown only
  // through its own random blob mask — some blobs take one copy, some the
  // other, some both (they blend), some neither (the noise vanishes
  // there). No frame-to-frame continuity, by design: new weights, new
  // noise. The white and purple variants share transforms and masks.
  function rebuildNoiseVariant(st) {
    const { W, H } = st;
    if (!st.noiseVariant) {
      st.noiseVariant = makeCanvas(W, H);
      st.noiseVariantP = makeCanvas(W, H);
      st.nvTmp = makeCanvas(W, H);
    }
    const rnd = (a, b) => a + Math.random() * (b - a);
    const seed = () => (1 + Math.random() * 0x7ffffffe) | 0;
    const mk = () => ({
      dx: rnd(-70, 70),
      dy: rnd(-16, 16),
      rot: rnd(-0.035, 0.035),
      fx: Math.random() < 0.5 ? -1 : 1,
      fy: Math.random() < 0.5 ? -1 : 1,
      a: rnd(0.75, 1),
      mask: makeFragMask(W, H, seed(), seed()),
    });
    const tA = mk();
    const tB = mk();
    const compose = (dst, imgA, imgB) => {
      const ctx = dst.getContext("2d");
      ctx.clearRect(0, 0, W, H);
      for (const [img, t] of [
        [imgA, tA],
        [imgB, tB],
      ]) {
        const tc = st.nvTmp.getContext("2d");
        tc.clearRect(0, 0, W, H);
        tc.save();
        // overscanned so the transform never exposes an edge
        tc.translate(W / 2 + t.dx, H / 2 + t.dy);
        tc.rotate(t.rot);
        tc.scale(1.12 * t.fx, 1.35 * t.fy);
        tc.drawImage(img, -W / 2, -H / 2);
        tc.restore();
        tc.globalCompositeOperation = "destination-in";
        tc.drawImage(t.mask, 0, 0);
        tc.globalCompositeOperation = "source-over";
        ctx.globalAlpha = t.a;
        ctx.drawImage(st.nvTmp, 0, 0);
      }
      ctx.globalAlpha = 1;
    };
    compose(st.noiseVariant, st.fp, st.fp2);
    compose(st.noiseVariantP, st.fpP, st.fpP2);
  }

  // the fake inference moment: the current prediction freezes under a dim
  // wash and an orange banner. Painting stays live the whole time.
  function drawGenerating() {
    const c = predRef.current;
    if (!c) return;
    const ctx = c.getContext("2d");
    ctx.fillStyle = "rgba(8,8,10,0.55)";
    ctx.fillRect(0, 0, c.width, c.height);
    const fs = Math.round(c.width / 44);
    ctx.font = `600 ${fs}px system-ui, sans-serif`;
    ctx.textAlign = "center";
    ctx.fillStyle = "#ff6a3d";
    ctx.fillText(
      "Generating prediction…",
      c.width / 2,
      c.height / 2 + fs * 0.35,
    );
    ctx.textAlign = "left";
  }

  // keep latest phase visible inside composite() without re-creating it
  const phaseRef = useRef(phase);
  phaseRef.current = phase;

  // a random real-ink pixel in a spot dense enough that a brush stamp
  // right on it scores as "on ink". The FIRST hint teaches reading order
  // (the first letter of the word); after that, hints roam the WHOLE
  // segment, pointing at ink the user hasn't labeled yet.
  function pickInkPoint(firstLetter) {
    const st = S.current;
    if (!st) return null;
    const { m, W, H } = st;
    const r = st.brush;
    const dense = (x, y) => {
      let ink = 0;
      let tot = 0;
      for (let yy = Math.max(0, y - r); yy <= Math.min(H - 1, y + r); yy += 3)
        for (let xx = Math.max(0, x - r); xx <= Math.min(W - 1, x + r); xx += 3) {
          if ((xx - x) ** 2 + (yy - y) ** 2 > r * r) continue;
          tot++;
          if (st.labelData[yy * W + xx] > 127) ink++;
        }
      return tot > 0 && ink / tot > 0.55;
    };
    if (firstLetter && (m.letters || []).length) {
      let idx = -1;
      m.letters.forEach((b, i) => {
        if (st.perLetter[i] / st.letterInk[i] > LETTER_LEARN_FRAC) return;
        if (idx < 0 || b.x < m.letters[idx].x) idx = i;
      });
      const b = m.letters[idx];
      if (b)
        for (let tries = 0; tries < 400; tries++) {
          const x = Math.round(b.x + Math.random() * b.w);
          const y = Math.round(b.y + Math.random() * b.h);
          if (st.labelData[y * W + x] > 127 && dense(x, y)) return { x, y };
        }
    }
    // how much of the brush disc around a spot is already labeled
    const labeledNear = (x, y) => {
      let lab = 0;
      let tot = 0;
      for (let yy = Math.max(0, y - r); yy <= Math.min(H - 1, y + r); yy += 3)
        for (let xx = Math.max(0, x - r); xx <= Math.min(W - 1, x + r); xx += 3) {
          tot++;
          if (st.labeledMask[yy * W + xx]) lab++;
        }
      return lab / Math.max(1, tot);
    };
    // roam: dense ink anywhere on the segment, away from existing labels
    for (let tries = 0; tries < 900; tries++) {
      const x = (Math.random() * W) | 0;
      const y = (Math.random() * H) | 0;
      if (
        st.labelData[y * W + x] > 127 &&
        dense(x, y) &&
        labeledNear(x, y) < 0.15
      )
        return { x, y };
    }
    // nearly everything is labeled already: any dense ink will do
    for (let tries = 0; tries < 400; tries++) {
      const x = (Math.random() * W) | 0;
      const y = (Math.random() * H) | 0;
      if (st.labelData[y * W + x] > 127 && dense(x, y)) return { x, y };
    }
    return null;
  }

  // stamp `count` real ink pixels (used by the auto-finish)
  function placeAutoStamps(count) {
    for (let k = 0; k < count; k++) {
      const p = pickInkPoint(false);
      if (p) stamp(p.x, p.y);
    }
  }

  // the hint points, it doesn't paint: a pulsing ring marks a spot of real
  // ink, and the user still does the finding themselves
  function hint() {
    const st = S.current;
    if (!st) return;
    const p = pickInkPoint(!st.hintedOnce);
    if (!p) return;
    st.hintedOnce = true;
    st.hintPt = p;
    drawRef.current.dataset.hint = `${Math.round(p.x)},${Math.round(p.y)}`;
    // "go there": center the view on the hint, so asking for one always
    // shows the ring — a no-op when the whole strip fits the pane
    // (desktop unzoomed); phones get a zoom bump so the ring is legible
    const v = st.view;
    if (st.minS > 1) v.s = Math.max(v.s, st.minS * 1.25);
    v.tx = st.dcW / 2 - p.x * v.s;
    v.ty = st.dcH / 2 - p.y * v.s;
    clampView(v);
    setHintOn(true);
    setStatus("See the pulsing ring? That texture is ink. Paint it.");
    redrawLeft();
  }

  function clearHint(found) {
    const st = S.current;
    if (!st?.hintPt) return;
    st.hintPt = null;
    delete drawRef.current.dataset.hint;
    setHintOn(false);
    if (found)
      setStatus("Found it. That texture, wherever it appears, is ink.");
  }

  // the demo opens with a hint already pulsing: the first move is shown,
  // not explained
  const autoHinted = useRef(false);
  useEffect(() => {
    if (phase !== "play" || autoHinted.current) return;
    autoHinted.current = true;
    hint();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [phase]);

  // keep the hint ring pulsing (skipped under prefers-reduced-motion)
  useEffect(() => {
    if (!hintOn || phase !== "play") return;
    if (window.matchMedia?.("(prefers-reduced-motion: reduce)").matches)
      return;
    const id = setInterval(redrawLeft, 90);
    return () => clearInterval(id);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [hintOn, phase]);

  // take EVERY label back out of the training set — strokes, reveal and
  // model reset together (the seg demo's "Clear this slice", for ink)
  function clearAll() {
    const st = S.current;
    if (!st || st.autoTimer) return;
    st.labeledMask.fill(0);
    st.labeled = 0;
    st.perLetter.fill(0);
    st.goodStamps = [];
    st.wrongStamps = [];
    st.revealDirty = true;
    st.strokes.getContext("2d").clearRect(0, 0, st.W, st.H);
    st.wrongWarnOn = false;
    st.corrHealedSaid = false;
    st.lastTaught = 0;
    st.generalizedDone = false;
    setGeneralized(false);
    clearHint(false);
    st.lastPct = 0;
    setLabeledPct(0);
    setStatus("All labels erased. The model resets at the next retrain.");
    redrawLeft();
    scheduleInference();
  }

  // hands-off ending for visitors who've tried a little and want to see
  // where it goes: the same labeling, just fast-forwarded
  function finishForMe() {
    const st = S.current;
    if (!st || st.autoTimer) return;
    clearHint(false);
    setTool("brush");
    toolRef.current = "brush";
    setAutoBusy(true);
    setStatus(
      st.wrongStamps.length
        ? "Fast-forwarding: erasing your bare-papyrus strokes, then labeling more ink."
        : "Fast-forwarding: more labels, same loop, just quicker.",
    );
    st.autoTimer = setInterval(() => {
      const cur = S.current;
      if (!cur) return;
      if (phaseRef.current === "done") {
        clearInterval(cur.autoTimer);
        cur.autoTimer = 0;
        setAutoBusy(false);
        return;
      }
      // strokes on bare papyrus can block the finish (the far-label
      // gate), so the fast-forward does what you'd have to do: erase them
      if (cur.wrongStamps.length) {
        toolRef.current = "eraser";
        for (let k = 0; k < 3 && cur.wrongStamps.length; k++) {
          const p = cur.wrongStamps[cur.wrongStamps.length - 1];
          stamp(p.x, p.y);
        }
        toolRef.current = "brush";
        return;
      }
      placeAutoStamps(4);
    }, 80);
  }

  useEffect(
    () => () => {
      if (S.current?.autoTimer) clearInterval(S.current.autoTimer);
      if (S.current?.infTimer) clearTimeout(S.current.infTimer);
      if (S.current?.pendingPaint) clearTimeout(S.current.pendingPaint.t);
    },
    [],
  );

  // test shortcut: /get_started?finish=1 runs the demo to completion and
  // opens the reveal, so the end states can be reviewed without playing
  const finishTest = useRef(false);
  useEffect(() => {
    if (!new URLSearchParams(window.location.search).has("finish")) return;
    finishTest.current = true;
    start().then(() => setTimeout(finishForMe, 400));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);
  useEffect(() => {
    if (phase === "done" && finishTest.current) {
      setReadOn(true);
      setOverlay(true);
    }
  }, [phase]);

  // on phones the payoff text renders below the fold, and hijacking the
  // scroll to it mid-gesture felt broken. Instead a fixed pill announces
  // the finish over whatever the user is doing (they can keep painting),
  // and tapping it takes them to the outro. CSS hides it on desktop,
  // where the outro is already on screen.
  const doneCueOn = phase === "done" && !readOn && !doneCueSeen;
  function openOutro() {
    setDoneCueSeen(true);
    setTimeout(
      () =>
        outroRef.current?.scrollIntoView({
          behavior: "smooth",
          block: "center",
        }),
      60,
    );
  }

  // pointer handlers (left canvas): one pointer paints, two pinch-zoom
  // (moving both fingers pans), right/middle drag pans on desktop
  function onPointerDown(e) {
    if (phaseRef.current !== "play" && phaseRef.current !== "done") return;
    const st = S.current;
    try {
      e.currentTarget.setPointerCapture(e.pointerId);
    } catch {
      /* pointer already gone — capture is best-effort */
    }
    const [cx, cy] = eventToImageXY(e, drawRef.current);
    if (e.pointerType === "mouse" && e.button !== 0) {
      e.preventDefault();
      st.panning = { x: cx, y: cy };
      return;
    }
    st.pointers.set(e.pointerId, { x: cx, y: cy });
    if (st.pointers.size === 2) {
      // the second finger of a pan/pinch: whatever the first finger
      // buffered was never a stroke — drop it undrawn
      if (st.pendingPaint) {
        clearTimeout(st.pendingPaint.t);
        st.pendingPaint = null;
      }
      // a LATE second finger (past the buffer window): the first finger
      // already painted a little — that was the start of a pan, not
      // labeling, so take the whole young stroke back out of the model
      const ss = st.strokeStart;
      if (ss && performance.now() - ss.t < 450) {
        for (const p of ss.fresh) {
          if (!st.labeledMask[p]) continue;
          st.labeledMask[p] = 0;
          st.labeled--;
          const xx = p % st.W;
          const yy = (p / st.W) | 0;
          (st.m.letters || []).forEach((b, i) => {
            if (xx >= b.x && xx < b.x + b.w && yy >= b.y && yy < b.y + b.h)
              st.perLetter[i] = Math.max(0, st.perLetter[i] - 1);
          });
        }
        if (st.goodStamps.length > ss.good) st.goodStamps.length = ss.good;
        if (st.wrongStamps.length > ss.wrong)
          st.wrongStamps.length = ss.wrong;
        if (ss.fresh.length || st.stampCount !== ss.count) {
          st.revealDirty = true;
          rebuildStrokes(st);
          redrawLeft();
          if (st.wrongWarnOn && st.wrongStamps.length === ss.wrong) {
            st.wrongWarnOn = false;
            setStatus("");
          }
          const pct = Math.round((st.labeled / st.inkTotal) * 100);
          if (pct !== st.lastPct) {
            st.lastPct = pct;
            setLabeledPct(pct);
          }
          scheduleInference();
        }
      }
      st.strokeStart = null;
      st.painting = false;
      const [p1, p2] = [...st.pointers.values()];
      const v = st.view;
      const mid = { x: (p1.x + p2.x) / 2, y: (p1.y + p2.y) / 2 };
      st.pinch0 = {
        d: Math.max(1, Math.hypot(p1.x - p2.x, p1.y - p2.y)),
        s: v.s,
        ix: (mid.x - v.tx) / v.s,
        iy: (mid.y - v.ty) / v.s,
      };
    } else if (e.pointerType === "touch") {
      // a finger can't commit to painting on contact: the second finger
      // of a two-finger pan lands a beat later, and stamping now would
      // draw a blot first. Buffer the points briefly and flush them if
      // no second finger shows up — the stroke paints retroactively.
      st.pendingPaint = {
        pts: [pointerToImg(e)],
        t: setTimeout(flushPendingPaint, 90),
      };
      st.strokeStart = {
        t: performance.now(),
        good: st.goodStamps.length,
        wrong: st.wrongStamps.length,
        count: st.stampCount,
        fresh: [],
      };
    } else {
      st.painting = true;
      stamp(...pointerToImg(e));
    }
  }

  function flushPendingPaint() {
    const st = S.current;
    if (!st?.pendingPaint) return;
    clearTimeout(st.pendingPaint.t);
    const pts = st.pendingPaint.pts;
    st.pendingPaint = null;
    st.painting = true;
    for (const p of pts) stamp(...p);
  }

  // redraw the visible strokes from the surviving stamp lists — used when
  // a young touch stroke is undone (it turned out to be a pan). Stamps
  // are thinned, so this is a hair lighter than the live drag, invisibly.
  function rebuildStrokes(st) {
    const ctx = st.strokes.getContext("2d");
    ctx.clearRect(0, 0, st.W, st.H);
    ctx.fillStyle = "rgba(229,80,43,0.45)";
    for (const p of [...st.goodStamps, ...st.wrongStamps]) {
      ctx.beginPath();
      ctx.arc(p.x, p.y, st.brush, 0, Math.PI * 2);
      ctx.fill();
    }
  }
  function onPointerMove(e) {
    const st = S.current;
    if (!st) return;
    if (st.panning) {
      const [cx, cy] = eventToImageXY(e, drawRef.current);
      const v = st.view;
      v.tx += cx - st.panning.x;
      v.ty += cy - st.panning.y;
      st.panning = { x: cx, y: cy };
      clampView(v);
      redrawLeft();
      return;
    }
    if (st.pointers.has(e.pointerId)) {
      const [cx, cy] = eventToImageXY(e, drawRef.current);
      st.pointers.set(e.pointerId, { x: cx, y: cy });
    }
    if (st.pendingPaint) {
      // still inside the is-it-a-pan window: collect, don't draw yet
      if (st.pointers.has(e.pointerId))
        st.pendingPaint.pts.push(pointerToImg(e));
      return;
    }
    if (st.pinch0 && st.pointers.size === 2) {
      const [p1, p2] = [...st.pointers.values()];
      const d = Math.max(1, Math.hypot(p1.x - p2.x, p1.y - p2.y));
      const mid = { x: (p1.x + p2.x) / 2, y: (p1.y + p2.y) / 2 };
      const v = st.view;
      v.s = st.pinch0.s * (d / st.pinch0.d);
      v.tx = mid.x - st.pinch0.ix * v.s;
      v.ty = mid.y - st.pinch0.iy * v.s;
      clampView(v);
      redrawLeft();
      return;
    }
    if (st.painting) stamp(...pointerToImg(e));
  }
  function onPointerUp(e) {
    const st = S.current;
    if (!st) return;
    st.panning = null;
    st.pointers.delete(e.pointerId);
    if (st.pendingPaint) {
      if (e.type === "pointercancel") {
        // the browser took the gesture — never paint from a cancel
        clearTimeout(st.pendingPaint.t);
        st.pendingPaint = null;
      } else {
        // finger up before the window closed: a quick tap still paints
        flushPendingPaint();
      }
    }
    if (st.pointers.size < 2) st.pinch0 = null;
    if (st.pointers.size === 0) {
      st.painting = false;
      st.strokeStart = null; // the stroke survived: it's real labeling
    }
  }

  // repaint the surface pane when the prediction overlay is toggled
  useEffect(() => {
    if (S.current && (phase === "play" || phase === "done")) redrawLeft();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [overlay]);

  const meterFill = Math.min(100, (labeledPct / (AREA_DONE_FRAC * 100)) * 100);

  return (
    <div className="vc-gs-demo" id="ink-demo">
      {phase === "idle" || phase === "loading" || phase === "error" ? (
        <div className="vc-gs-demo__poster">
          <img
            src={base + "papyrus.webp"}
            alt="X-ray surface render of carbonized papyrus with faint ink traces"
            loading="lazy"
            decoding="async"
          />
          <div className="vc-gs-demo__poster-overlay">
            <p>
              A surface from inside the same generous scroll, as the X-ray
              sees it. The ink survives as the faint cracked texture. Label
              what you can see, and a model trains on your strokes to
              predict the rest.
            </p>
            <button
              className="vc-btn"
              onClick={start}
              disabled={phase === "loading"}
            >
              {phase === "loading"
                ? "Loading the scroll…"
                : phase === "error"
                  ? "Retry"
                  : "Find the ink"}
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
        {/* the strip is one wide line of text: stack the panes */}
        <div className="vc-gs-demo__panes vc-gs-demo__panes--stack">
          <figure className="vc-gs-demo__pane">
            <canvas
              ref={drawRef}
              className="vc-gs-demo__canvas vc-gs-demo__canvas--draw"
              onPointerDown={onPointerDown}
              onPointerMove={onPointerMove}
              onPointerUp={onPointerUp}
              onPointerCancel={onPointerUp}
              onContextMenu={(e) => e.preventDefault()}
            />
            <figcaption>
              Papyrus surface: paint the ink{" "}
              <span className="vc-gs-dim">
                (bright cracked patches) ·{" "}
                {coarse
                  ? "pinch to zoom, drag with two fingers to pan"
                  : "scroll to zoom, drag with the right mouse button to pan"}
              </span>
            </figcaption>
          </figure>
          <figure className="vc-gs-demo__pane">
            <canvas ref={predRef} className="vc-gs-demo__canvas" />
            <figcaption>
              Model prediction. It retrains on your labels every few
              strokes.
            </figcaption>
          </figure>
        </div>

        <div className="vc-gs-demo__bar">
          <div className="vc-gs-demo__tools" role="group" aria-label="Tools">
            <button
              className={`vc-gs-tool ${tool === "brush" ? "vc-gs-tool--on" : ""}`}
              onClick={() => setTool("brush")}
            >
              Ink brush
            </button>
            <button
              className={`vc-gs-tool ${tool === "eraser" ? "vc-gs-tool--on" : ""}`}
              onClick={() => setTool("eraser")}
            >
              Erase
            </button>
            <button className="vc-gs-tool" onClick={clearAll}>
              Clear labels
            </button>
            <button
              className={`vc-gs-tool ${overlay ? "vc-gs-tool--on" : ""}`}
              onClick={() => setOverlay((v) => !v)}
              aria-pressed={overlay}
            >
              Overlay prediction
            </button>
            <button className="vc-gs-tool" onClick={hint}>
              Hint
            </button>
            {/* only offered once they've genuinely tried: the point of the
                demo is doing it, not watching it */}
            {phase === "play" && labeledPct >= 2 && !autoBusy && (
              <button className="vc-gs-tool" onClick={finishForMe}>
                Finish it for me
              </button>
            )}
          </div>
          <div
            className="vc-gs-meter"
            role="progressbar"
            aria-valuenow={labeledPct}
            aria-valuemin={0}
            aria-valuemax={100}
            aria-label="Ink labeled"
          >
            <div
              className="vc-gs-meter__fill"
              style={{ width: `${meterFill}%` }}
            />
            <span className="vc-gs-meter__label">
              ink labeled {labeledPct}%
            </span>
          </div>
        </div>
        <p className="vc-gs-demo__status" aria-live="polite" ref={statusRef}>
          {status}
        </p>

        {doneCueOn && (
          <div className="vc-gs-donecue">
            <button className="vc-btn" onClick={openOutro}>
              ✓ Your model converged — read what you found
            </button>
          </div>
        )}

        {phase === "done" && (
          <div className="vc-gs-demo__outro" ref={outroRef}>
            <p className="vc-gs-demo__win">
              You labeled a fraction of the ink and the model read the
              rest. That loop, scaled up ten-thousand-fold, is how the
              first scroll was read. Those bright patches were never
              random: you've been labeling ancient Greek this whole time.
            </p>
            {readOn ? (
              <div className="vc-gs-read">
                <div className="vc-gs-read__words">
                  <span className="vc-gs-read__word">
                    ΠΟΡΦΥΡΑϹ<small>porphyras · purple</small>
                  </span>
                </div>
                <p className="vc-gs-read__note">
                  This patch of papyrus is famous. In 2023 it gave up the
                  first word ever read from inside a sealed scroll, and the
                  two students who found it won the{" "}
                  <a href="/firstletters">First Letters prize</a>. You just
                  retraced their steps on the real data. You can do it for
                  real. Most scanned scrolls have never been read, and
                  finding ten letters inside one of them{" "}
                  <a href="/prizes#first-letters-prizes">
                    pays $50,000
                  </a>
                  .
                </p>
              </div>
            ) : (
              <div className="vc-gs-demo__reward vc-gs-spiral__invite">
                <button
                  className="vc-btn"
                  onClick={() => {
                    setReadOn(true);
                    setOverlay(true);
                  }}
                >
                  Read what you found
                </button>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

// is there real ink within R of (x, y)? Splits bad labels into edge slop
// (near a letter — normal labeling) and deliberate painting out on the
// open papyrus (far from all ink) — only the far kind can block the
// finish. A handful of hits is required so stray label speckle can't
// rescue a genuinely bare stamp.
function inkNear(labelData, W, H, x, y, R) {
  const x0 = Math.max(0, Math.round(x - R));
  const x1 = Math.min(W - 1, Math.round(x + R));
  const y0 = Math.max(0, Math.round(y - R));
  const y1 = Math.min(H - 1, Math.round(y + R));
  const R2 = R * R;
  let hits = 0;
  for (let yy = y0; yy <= y1; yy += 2)
    for (let xx = x0; xx <= x1; xx += 2) {
      const dx = xx - x;
      const dy = yy - y;
      if (dx * dx + dy * dy > R2) continue;
      if (labelData[yy * W + xx] > 127 && ++hits > 8) return true;
    }
  return false;
}

// the soft evidence halo one correct stamp contributes to the reveal —
// used both live (as you paint) and when the reveal is rebuilt from the
// surviving stamps after an erase, so the two must stay identical
function paintHalo(rctx, x, y, r) {
  const rr = r * 6;
  const g = rctx.createRadialGradient(x, y, r * 0.7, x, y, rr);
  g.addColorStop(0, "rgba(255,255,255,0.85)");
  g.addColorStop(0.5, "rgba(255,255,255,0.4)");
  g.addColorStop(1, "rgba(255,255,255,0)");
  rctx.fillStyle = g;
  rctx.beginPath();
  rctx.arc(x, y, rr, 0, Math.PI * 2);
  rctx.fill();
}

// seeded blurred random field at a given cell resolution, as raw 0-255
// luminance sampled at W x H
function blurredNoiseField(W, H, gw, gh, seed0) {
  const small = makeCanvas(gw, gh);
  const sctx = small.getContext("2d");
  const id = sctx.createImageData(gw, gh);
  let seed = seed0;
  for (let i = 0; i < gw * gh; i++) {
    // Seeded xorshift: irregular but deterministic, with no diagonal lattice.
    seed ^= seed << 13;
    seed ^= seed >>> 17;
    seed ^= seed << 5;
    const v = seed >>> 24;
    id.data[i * 4] = id.data[i * 4 + 1] = id.data[i * 4 + 2] = v;
    id.data[i * 4 + 3] = 255;
  }
  sctx.putImageData(id, 0, 0);
  const big = makeCanvas(W, H);
  const bctx = big.getContext("2d");
  bctx.imageSmoothingEnabled = true;
  bctx.drawImage(small, 0, 0, W, H);
  return bctx.getImageData(0, 0, W, H).data;
}

// two-octave blurred random field, normalized 0..1 — the "order" in which
// unlabeled regions fade in as the model generalizes. The coarse octave
// decides which regions come first; the fine one roughens their edges so
// emerging patches read as prediction fragments, not soft blobs.
function makeOrderField(W, H) {
  // Keep cells roughly square for this very wide strip. A square 24x24
  // field stretched to the canvas produced obvious horizontal bands.
  const coarse = blurredNoiseField(W, H, 64, 24, 0x6d2b79f5);
  const fine = blurredNoiseField(W, H, 208, 48, 0x9e3779b9);
  const f = new Float32Array(W * H);
  let mn = Infinity;
  let mx = -Infinity;
  for (let i = 0; i < W * H; i++) {
    const v = coarse[i * 4] * 0.72 + fine[i * 4] * 0.28;
    f[i] = v;
    if (v < mn) mn = v;
    if (v > mx) mx = v;
  }
  const rng = Math.max(1, mx - mn);
  for (let i = 0; i < W * H; i++) f[i] = (f[i] - mn) / rng;
  return f;
}

// soft-thresholded two-octave random field: the mask that breaks early
// local reveals into irregular clusters of prediction fragments. The
// coarse octave opens and closes whole lobes of the halo (so its round
// footprint disappears); the fine one shreds the lobe edges.
function makeFragMask(W, H, seedA, seedB) {
  const coarse = blurredNoiseField(W, H, 44, 13, seedA);
  const fine = blurredNoiseField(W, H, 130, 30, seedB);
  const c = makeCanvas(W, H);
  const ctx = c.getContext("2d");
  const id = ctx.createImageData(W, H);
  let mn = Infinity;
  let mx = -Infinity;
  const v = new Float32Array(W * H);
  for (let i = 0; i < W * H; i++) {
    v[i] = coarse[i * 4] * 0.5 + fine[i * 4] * 0.5;
    if (v[i] < mn) mn = v[i];
    if (v[i] > mx) mx = v[i];
  }
  const rng = Math.max(1, mx - mn);
  for (let i = 0; i < W * H; i++) {
    const x = (v[i] - mn) / rng;
    // smoothstep window: ~half the field passes, with soft edges
    const t = Math.min(1, Math.max(0, (x - 0.42) / 0.3));
    const a = t * t * (3 - 2 * t);
    const j = i * 4;
    id.data[j] = id.data[j + 1] = id.data[j + 2] = 255;
    id.data[j + 3] = a * 255;
  }
  ctx.putImageData(id, 0, 0);
  return c;
}
