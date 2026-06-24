// AtlasRendererProvider.js
//
// Provides ONE shared WebGL renderer + a single full-screen <canvas> that draws
// every card's 3D model via scissored viewports (the "shared canvas" technique
// ported from the reference index.html grid renderer). Cards register a DOM
// element + a mesh manifest entry through the context's imperative API; the
// provider lazy-loads each PLY on scroll, spins it, and renders it into the
// screen-space rectangle of its card.
//
// Why a single renderer + single canvas: creating one WebGLRenderer per card
// would blow the browser's WebGL-context budget (~16). Here every model shares
// the same context and we just move the viewport/scissor box per frame.
//
// SSR-safe: three is imported only inside the client-only mount effect; nothing
// touches window/document/three during render.

import React, { createContext, useEffect, useRef, useState } from 'react';
import { buildScene, meshUrl, disposeScene } from './threeScene';

// Context exposes a STABLE imperative API (see below). Default null so that
// consumers rendered outside a provider (or during SSR) can no-op gracefully.
export const AtlasRendererContext = createContext(null);

export default function AtlasRendererProvider({ children }) {
  // The shared full-screen canvas element.
  const canvasRef = useRef(null);

  // Mutable registry of cards. id -> { id, el, mesh, scene, group, camera, loading }.
  // Deliberately NOT React state: it's mutated every frame and on every
  // scroll-triggered load, and must never trigger re-renders.
  const registryRef = useRef(new Map());

  // Holds three.js singletons created on mount (renderer, loader, observer, …)
  // plus bookkeeping. Populated inside the mount effect.
  const glRef = useRef(null);

  // rAF handle so cleanup can cancel the loop.
  const rafRef = useRef(0);

  // Guard so the renderer is created exactly once even under React 18
  // StrictMode's double mount/unmount in development.
  const initedRef = useRef(false);

  // `paused` is the ONLY piece of React state, so a pause button can reflect
  // it. Default to true when the user prefers reduced motion.
  const [paused, setPausedState] = useState(() => {
    if (typeof window === 'undefined' || !window.matchMedia) return false;
    return window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  });

  // Keep a ref mirror of `paused` so the rAF loop reads the latest value
  // without needing to be torn down / recreated when it changes.
  const pausedRef = useRef(paused);
  pausedRef.current = paused;

  // The stable imperative API handed out via context. Created once (useRef) so
  // its identity never changes — consumers can safely list it in deps without
  // re-running effects, and it works before the mount effect has populated glRef.
  const apiRef = useRef(null);
  if (apiRef.current === null) {
    const setPaused = (next) => {
      setPausedState((prev) => {
        const value = typeof next === 'function' ? next(prev) : !!next;
        pausedRef.current = value;
        return value;
      });
    };

    // Try to lazy-load the PLY for an entry once its card scrolls into view.
    const tryLoad = (entry) => {
      const gl = glRef.current;
      if (!gl || !gl.loader || !gl.THREE) return; // renderer not ready yet
      if (entry.loading || entry.scene || !entry.mesh) return;
      entry.loading = true;
      gl.loader.load(
        meshUrl(entry.mesh.file),
        (geom) => {
          // Guard: card may have unregistered while the PLY was downloading.
          if (!registryRef.current.has(entry.id)) {
            geom.dispose && geom.dispose();
            return;
          }
          const { scene, group, camera } = buildScene(gl.THREE, geom, entry.mesh);
          Object.assign(entry, { scene, group, camera });
          // Remove the "loading 3D…" placeholder once the model is ready.
          const ph = entry.el && entry.el.querySelector('.ph');
          if (ph) ph.remove();
        },
        undefined,
        () => {
          entry.loading = false;
          const ph = entry.el && entry.el.querySelector('.ph');
          if (ph) ph.textContent = '3D load failed';
        }
      );
    };

    apiRef.current = {
      register(id, el, mesh) {
        if (!el) return;
        const entry = {
          id,
          el,
          mesh,
          scene: null,
          group: null,
          camera: null,
          loading: false,
        };
        registryRef.current.set(id, entry);
        const gl = glRef.current;
        if (gl && gl.io) gl.io.observe(el);
      },
      unregister(id) {
        const entry = registryRef.current.get(id);
        if (!entry) return;
        const gl = glRef.current;
        if (gl && gl.io && entry.el) gl.io.unobserve(entry.el);
        if (entry.scene) disposeScene(entry.scene);
        registryRef.current.delete(id);
      },
      setPaused,
      get paused() {
        return pausedRef.current;
      },
      // Internal: used by the IntersectionObserver callback created in the
      // mount effect to drive lazy loading. Not part of the public contract.
      _tryLoad: tryLoad,
    };
  }

  // ---- Single client-only mount effect: create the renderer & loop. ----
  useEffect(() => {
    // StrictMode runs effects twice in dev; create GL resources only once.
    if (initedRef.current) return undefined;
    initedRef.current = true;

    let cancelled = false;

    (async () => {
      const THREE = await import('three');
      const { PLYLoader } = await import('three/examples/jsm/loaders/PLYLoader.js');
      if (cancelled) return; // unmounted before three finished loading

      const canvas = canvasRef.current;
      if (!canvas) return;

      const renderer = new THREE.WebGLRenderer({
        canvas,
        antialias: true,
        alpha: true,
      });
      renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
      renderer.setClearColor(0x000000, 0);
      renderer.autoClear = false; // we clear once per frame, then scissor-render each card

      const resize = () => renderer.setSize(window.innerWidth, window.innerHeight, false);
      window.addEventListener('resize', resize);
      resize();

      const loader = new PLYLoader();

      // ONE IntersectionObserver drives lazy loading for all cards.
      const io = new IntersectionObserver(
        (entries) => {
          entries.forEach((e) => {
            if (!e.isIntersecting) return;
            // Find the registry entry whose element matches.
            for (const entry of registryRef.current.values()) {
              if (entry.el === e.target) {
                apiRef.current._tryLoad(entry);
                break;
              }
            }
          });
        },
        { rootMargin: '300px' }
      );

      glRef.current = { THREE, renderer, loader, io, resize };

      // Any cards that registered before three finished loading still need to
      // be observed.
      for (const entry of registryRef.current.values()) {
        io.observe(entry.el);
      }

      // ---- rAF frame loop (ported from reference index.html ~289-305). ----
      let last = performance.now();
      const frame = (now) => {
        const dt = (now - last) / 1000;
        last = now;
        renderer.clear();
        const w = window.innerWidth;
        const h = window.innerHeight;
        for (const entry of registryRef.current.values()) {
          if (!entry.scene) continue;
          const rect = entry.el.getBoundingClientRect();
          // Cull cards fully outside the viewport.
          if (
            rect.bottom < 0 ||
            rect.top > h ||
            rect.right < 0 ||
            rect.left > w
          ) {
            continue;
          }
          if (!pausedRef.current) entry.group.rotation.y += dt * 0.6;
          // WebGL's Y origin is bottom-left; convert from top-left rect.
          const y = h - rect.bottom;
          renderer.setViewport(rect.left, y, rect.width, rect.height);
          renderer.setScissor(rect.left, y, rect.width, rect.height);
          renderer.setScissorTest(true);
          entry.camera.aspect = rect.width / rect.height;
          entry.camera.updateProjectionMatrix();
          renderer.render(entry.scene, entry.camera);
        }
        rafRef.current = requestAnimationFrame(frame);
      };
      rafRef.current = requestAnimationFrame(frame);
    })();

    // ---- Cleanup on provider unmount. ----
    return () => {
      cancelled = true;
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
      rafRef.current = 0;
      const gl = glRef.current;
      if (gl) {
        if (gl.io) gl.io.disconnect();
        if (gl.resize) window.removeEventListener('resize', gl.resize);
        // Dispose any scenes still in the registry.
        for (const entry of registryRef.current.values()) {
          if (entry.scene) disposeScene(entry.scene);
        }
        if (gl.renderer) {
          // forceContextLoss() prevents leaking WebGL contexts across SPA
          // navigations (browsers cap simultaneous contexts).
          gl.renderer.dispose();
          gl.renderer.forceContextLoss();
        }
      }
      glRef.current = null;
      registryRef.current.clear();
      initedRef.current = false;
    };
  }, []);

  return (
    <AtlasRendererContext.Provider value={apiRef.current}>
      {/* Shared full-screen canvas. Styled globally (fixed, full-screen,
          pointer-events:none, z-index above content). We only create it
          with the className + ref here. */}
      <canvas ref={canvasRef} className="atlas-canvas" aria-hidden="true" />
      {children}
    </AtlasRendererContext.Provider>
  );
}
