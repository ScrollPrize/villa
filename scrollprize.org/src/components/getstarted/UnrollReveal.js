import React, { useEffect, useRef, useState } from "react";
import useBaseUrl from "@docusaurus/useBaseUrl";
import { loadImage } from "./demoUtils";

/*
 * "Unroll it" — the seg demo's completion reward, with REAL geometry:
 * seg/unroll.json holds a decimated grid of the porphyras strip's tifxyz
 * (absolute volume coordinates), and the texture is the ink demo's
 * papyrus.webp — the same flattened reference crop, usually already in
 * the browser cache.
 *
 * The unroll is PHYSICAL, a rolling contact: at progress p, the sheet up
 * to arc length p·L already lies flat in its final position, and the
 * remainder keeps its true curled tifxyz shape, rigidly rotated so its
 * tangent at the peel point continues the flat strip. The roll travels
 * along the strip as it lays the sheet down, passing through its real
 * volume shape on the way. When the strip is flat, the ink model's
 * prediction fades in over it, tinted purple: ΠΟΡΦΥΡΑϹ.
 *
 * The z axis is negated in the projection, same convention as the
 * turntable sprite: this segment's text-up direction is -z.
 */

const CW = 800;
const CH = 460;
const ROWS = 9; // animation grid (subsampled from the JSON grid)
const CURL_MS = 1400;
const UNROLL_MS = 3600;
const INK_MS = 900;
const FEATHER = 0.14; // arc fraction over which the sheet lifts off the strip
const PRED_TINT = "#8d00ff"; // porphyras

export default function UnrollReveal() {
  const segBase = useBaseUrl("/get-started/seg/");
  const inkBase = useBaseUrl("/get-started/ink/");
  const canvasRef = useRef(null);
  const S = useRef(null);
  const [stage, setStage] = useState(0); // 0 curl | 1 unroll | 2 read
  const [failed, setFailed] = useState(false);

  async function setup() {
    const res = await fetch(`${segBase}unroll.json`);
    if (!res.ok) throw new Error(`unroll.json ${res.status}`);
    const grid = await res.json();
    const [tex, pred] = await Promise.all([
      loadImage(inkBase + "papyrus.webp"),
      loadImage(inkBase + "prediction.webp"),
    ]);
    // the prediction tinted purple for the final fade-in
    const predP = document.createElement("canvas");
    predP.width = pred.naturalWidth;
    predP.height = pred.naturalHeight;
    const ppctx = predP.getContext("2d");
    ppctx.drawImage(pred, 0, 0);
    ppctx.globalCompositeOperation = "multiply";
    ppctx.fillStyle = PRED_TINT;
    ppctx.fillRect(0, 0, predP.width, predP.height);

    const { rows, cols, center, flatAspect, pts } = grid;
    const ri = Array.from({ length: ROWS }, (_, k) =>
      Math.round((k * (rows - 1)) / (ROWS - 1)),
    );
    // centered 3D points, subsampled rows: P3[r][c] = [dx, dy, dz]
    const P3 = ri.map((r) =>
      Array.from({ length: cols }, (_, c) => {
        const [x, y, z] = pts[r * cols + c];
        return [x - center[0], y - center[1], z - center[2]];
      }),
    );
    // flat targets: a vertical plane facing the camera at theta = 0,
    // text upright (v down = -dz)
    const h0 = P3[0][Math.floor(cols / 2)];
    const h1 = P3[ROWS - 1][Math.floor(cols / 2)];
    const Hg = Math.hypot(h0[0] - h1[0], h0[1] - h1[1], h0[2] - h1[2]);
    const L = flatAspect * Hg;
    const PF = ri.map((_, k) =>
      Array.from({ length: cols }, (_, c) => [
        (c / (cols - 1) - 0.5) * L,
        0,
        (0.5 - k / (ROWS - 1)) * Hg,
      ]),
    );
    // per-column arc length (mean over rows), column means, xy tangents
    const arc = new Float64Array(cols);
    for (let c = 1; c < cols; c++) {
      let d = 0;
      for (let r = 0; r < ROWS; r++)
        d += Math.hypot(
          P3[r][c][0] - P3[r][c - 1][0],
          P3[r][c][1] - P3[r][c - 1][1],
          P3[r][c][2] - P3[r][c - 1][2],
        );
      arc[c] = arc[c - 1] + d / ROWS;
    }
    const mid = Array.from({ length: cols }, (_, c) => {
      let mx = 0;
      let my = 0;
      for (let r = 0; r < ROWS; r++) {
        mx += P3[r][c][0];
        my += P3[r][c][1];
      }
      return [mx / ROWS, my / ROWS];
    });
    const tan = Array.from({ length: cols }, (_, c) => {
      const a = mid[Math.max(0, c - 1)];
      const b = mid[Math.min(cols - 1, c + 1)];
      const n = Math.hypot(b[0] - a[0], b[1] - a[1]) || 1;
      return [(b[0] - a[0]) / n, (b[1] - a[1]) / n];
    });

    let rmax = 0;
    let zmax = 0;
    for (const row of P3)
      for (const [dx, dy, dz] of row) {
        rmax = Math.max(rmax, Math.hypot(dx, dy));
        zmax = Math.max(zmax, Math.abs(dz));
      }
    const phi0 = (26 * Math.PI) / 180;
    const s3d = Math.min(
      (CW / 2 - 14) / rmax,
      (CH / 2 - 14) / (zmax * Math.cos(phi0) + rmax * Math.sin(phi0)),
    );
    const sflat = Math.min((CW - 24) / L, (CH - 24) / Hg);
    S.current = {
      P3,
      PF,
      cols,
      tex,
      predP,
      arc,
      mid,
      tan,
      L,
      Hg,
      phi0,
      s3d,
      sflat,
      t0: performance.now(),
      raf: 0,
      finished: false,
      // texture x range of the winding the user actually traced (opphi
      // within the porphyras strip) — warm-tinted during the curl phase
      traceU: [0.154, 0.507],
    };
  }

  function frame(now) {
    const st = S.current;
    const c = canvasRef.current;
    if (!st || !c) return;
    const t = now - st.t0;
    let prog = 0; // unroll progress
    let inkA = 0; // prediction fade
    let theta;
    const theta0 = 0.9 + CURL_MS * 0.00012;
    if (t < CURL_MS) {
      theta = 0.9 + t * 0.00012;
      setStage(0); // React bails out when unchanged
    } else if (t < CURL_MS + UNROLL_MS) {
      const k = (t - CURL_MS) / UNROLL_MS;
      prog = k * k * (3 - 2 * k);
      // the camera settles head-on early; the peel does the motion
      const kc = Math.min(1, prog / 0.35);
      theta = theta0 * (1 - kc * kc * (3 - 2 * kc));
      setStage(1);
    } else {
      prog = 1;
      theta = 0;
      inkA = Math.min(1, (t - CURL_MS - UNROLL_MS) / INK_MS);
      setStage(2);
    }
    draw(theta, prog, inkA);
    if (inkA >= 1) {
      st.finished = true;
      st.raf = 0;
      return;
    }
    st.raf = requestAnimationFrame(frame);
  }

  const smooth = (k) => k * k * (3 - 2 * k);

  function draw(theta, prog, inkA) {
    const st = S.current;
    const c = canvasRef.current;
    const ctx = c.getContext("2d");
    const { P3, PF, cols, tex, arc, mid, tan, L } = st;
    const cos = Math.cos(theta);
    const sin = Math.sin(theta);
    const phi = st.phi0 * (1 - prog);
    const cphi = Math.cos(phi);
    const sphi = Math.sin(phi);
    const scale = st.s3d + (st.sflat - st.s3d) * prog;
    const texW = tex.naturalWidth;
    const texH = tex.naturalHeight;

    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.fillStyle = "#0b0c0e";
    ctx.fillRect(0, 0, CW, CH);

    // rolling contact at arc position sp: rotation R aligns the sheet's
    // tangent there with +x, translation puts it on the flat strip
    const total = arc[cols - 1];
    const sp = prog * total;
    let cp = 0;
    while (cp < cols - 1 && arc[cp + 1] < sp) cp++;
    const f =
      arc[cp + 1] > arc[cp]
        ? Math.min(1, Math.max(0, (sp - arc[cp]) / (arc[cp + 1] - arc[cp])))
        : 0;
    const lerp2 = (a, b) => [
      a[0] + (b[0] - a[0]) * f,
      a[1] + (b[1] - a[1]) * f,
    ];
    const Mc = lerp2(mid[cp], mid[Math.min(cols - 1, cp + 1)]);
    const Tc = lerp2(tan[cp], tan[Math.min(cols - 1, cp + 1)]);
    const tn = Math.hypot(Tc[0], Tc[1]) || 1;
    const ca = Tc[0] / tn; // cos/sin of the tangent's angle; rotating by
    const sa = Tc[1] / tn; // -angle maps the tangent onto +x
    const Ax = (sp / total - 0.5) * L; // flat anchor x (arc-true position)
    const F = Math.max(1e-6, FEATHER * total);
    // during the first stretch the whole sheet eases from its resting
    // pose into the rolling frame (and the camera turns to meet it)
    const intro = smooth(Math.min(1, prog / 0.18));

    // morph + project every grid point
    const proj = P3.map((row, r) =>
      row.map((p, cc) => {
        let dx;
        let dy;
        let dz;
        const flat = PF[r][cc];
        if (arc[cc] <= sp) {
          [dx, dy, dz] = flat;
        } else {
          const e = smooth(Math.min(1, (arc[cc] - sp) / F));
          // rigid: rotate the curled remainder about the peel point onto
          // the strip's end (tangent-continuous = rolling without slipping)
          const qx = p[0] - Mc[0];
          const qy = p[1] - Mc[1];
          const rx = Ax + qx * ca + qy * sa;
          const ry = -qx * sa + qy * ca;
          dx = flat[0] + (rx - flat[0]) * e;
          dy = flat[1] + (ry - flat[1]) * e;
          dz = flat[2] + (p[2] - flat[2]) * e;
        }
        if (intro < 1) {
          dx = p[0] + (dx - p[0]) * intro;
          dy = p[1] + (dy - p[1]) * intro;
          dz = p[2] + (dz - p[2]) * intro;
        }
        const depth = -dx * sin + dy * cos;
        return {
          x: CW / 2 + (dx * cos + dy * sin) * scale,
          y: CH / 2 + (-dz * cphi - depth * sphi) * scale,
          d: depth,
        };
      }),
    );

    // painter's algorithm over quads
    const quads = [];
    for (let r = 0; r + 1 < proj.length; r++)
      for (let cc = 0; cc + 1 < cols; cc++) {
        const q = [proj[r][cc], proj[r][cc + 1], proj[r + 1][cc + 1], proj[r + 1][cc]];
        quads.push({ r, c: cc, q, d: (q[0].d + q[1].d + q[2].d + q[3].d) / 4 });
      }
    quads.sort((a, b) => b.d - a.d); // far first

    const v0 = (r) => (r / (ROWS - 1)) * (texH - 1);
    const u0 = (cc) => (cc / (cols - 1)) * (texW - 1);
    for (const { c: cc, r, q } of quads) {
      const tx0 = u0(cc);
      const tx1 = u0(cc + 1);
      const ty0 = v0(r);
      const ty1 = v0(r + 1);
      texTri(ctx, tex, q[0], q[1], q[3], tx0, ty0, tx1, ty0, tx0, ty1);
      texTri(ctx, tex, q[2], q[1], q[3], tx1, ty1, tx1, ty0, tx0, ty1);
      // simple screen-space shading: quads turning away from the camera
      // darken, so the curl reads as 3D
      const ax = q[1].x - q[0].x;
      const ay = q[1].y - q[0].y;
      const bx = q[3].x - q[0].x;
      const by = q[3].y - q[0].y;
      const area = ax * by - ay * bx;
      // full = the quad's screen area if it faced the camera head-on
      const full =
        (st.L / (cols - 1)) * (st.Hg / (ROWS - 1)) * scale * scale;
      const b = Math.min(1, Math.abs(area) / Math.max(1e-6, full));
      const dark = Math.max(0, Math.min(0.5, (1 - b) * 0.55));
      const uMid = (cc + 0.5) / (cols - 1);
      const warm =
        uMid >= st.traceU[0] && uMid <= st.traceU[1]
          ? 0.16 * (1 - prog)
          : 0;
      if (dark > 0.02 || warm > 0.01) {
        ctx.beginPath();
        ctx.moveTo(q[0].x, q[0].y);
        ctx.lineTo(q[1].x, q[1].y);
        ctx.lineTo(q[2].x, q[2].y);
        ctx.lineTo(q[3].x, q[3].y);
        ctx.closePath();
        if (dark > 0.02) {
          ctx.fillStyle = `rgba(0,0,0,${dark})`;
          ctx.fill();
        }
        if (warm > 0.01) {
          ctx.fillStyle = `rgba(229,80,43,${warm})`;
          ctx.fill();
        }
      }
    }

    // the ink model's reading fades in over the flat sheet, in purple
    if (inkA > 0) {
      const a = proj[0][0];
      const b2 = proj[ROWS - 1][cols - 1];
      ctx.globalCompositeOperation = "screen";
      ctx.globalAlpha = inkA * 0.95;
      ctx.drawImage(st.predP, a.x, a.y, b2.x - a.x, b2.y - a.y);
      ctx.globalAlpha = 1;
      ctx.globalCompositeOperation = "source-over";
    }
  }

  function play() {
    const st = S.current;
    if (!st) return;
    if (st.raf) cancelAnimationFrame(st.raf);
    st.finished = false;
    if (window.matchMedia?.("(prefers-reduced-motion: reduce)").matches) {
      setStage(2);
      draw(0, 1, 1);
      return;
    }
    st.t0 = performance.now();
    st.raf = requestAnimationFrame(frame);
  }

  useEffect(() => {
    let alive = true;
    setup()
      .then(() => {
        if (!alive) return;
        play();
      })
      .catch((e) => {
        console.error(e);
        setFailed(true);
      });
    return () => {
      alive = false;
      if (S.current?.raf) cancelAnimationFrame(S.current.raf);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  if (failed)
    return (
      <p className="vc-gs-demo__err">
        Couldn't load the unroll data. The{" "}
        <a href="/tutorial_spiral">tutorials</a> show the real pipeline.
      </p>
    );

  const captions = [
    "Your sheet, still curled, exactly as it sits inside the scroll.",
    "It peels off the roll and lies down flat…",
    "…and the ink model reads it. ΠΟΡΦΥΡΑϹ. Purple. The first word ever read from inside a sealed scroll.",
  ];

  return (
    <div className="vc-gs-spiral">
      <canvas
        ref={canvasRef}
        width={CW}
        height={CH}
        className="vc-gs-spiral__canvas"
        role="img"
        aria-label="The traced sheet unrolling from its curled position in the scroll into a flat strip that reads porphyras"
      />
      <div className="vc-gs-spiral__bar">
        <p className="vc-gs-spiral__caption" aria-live="polite">
          {captions[stage]}
        </p>
        <button className="vc-gs-tool" onClick={play}>
          Replay
        </button>
      </div>
    </div>
  );
}

// affine texture-map one triangle: (x/y screen) <- (u/v texture px)
function texTri(ctx, img, p0, p1, p2, u0, v0, u1, v1, u2, v2) {
  const d = (u1 - u0) * (v2 - v0) - (u2 - u0) * (v1 - v0);
  if (Math.abs(d) < 1e-9) return;
  const a = ((p1.x - p0.x) * (v2 - v0) - (p2.x - p0.x) * (v1 - v0)) / d;
  const b = ((p2.x - p0.x) * (u1 - u0) - (p1.x - p0.x) * (u2 - u0)) / d;
  const c = ((p1.y - p0.y) * (v2 - v0) - (p2.y - p0.y) * (v1 - v0)) / d;
  const e = ((p2.y - p0.y) * (u1 - u0) - (p1.y - p0.y) * (u2 - u0)) / d;
  const f = p0.x - a * u0 - b * v0;
  const g = p0.y - c * u0 - e * v0;
  ctx.save();
  // inflate the clip slightly so quad seams don't show
  const cx = (p0.x + p1.x + p2.x) / 3;
  const cy = (p0.y + p1.y + p2.y) / 3;
  const grow = (p) => [cx + (p.x - cx) * 1.03, cy + (p.y - cy) * 1.03];
  ctx.beginPath();
  ctx.moveTo(...grow(p0));
  ctx.lineTo(...grow(p1));
  ctx.lineTo(...grow(p2));
  ctx.closePath();
  ctx.clip();
  ctx.setTransform(a, c, b, e, f, g);
  ctx.drawImage(img, 0, 0);
  ctx.restore();
}
