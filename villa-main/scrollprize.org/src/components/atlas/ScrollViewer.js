// ScrollViewer.js
//
// Self-contained INTERACTIVE 3D viewer for the detail page. Unlike the shared
// grid canvas (AtlasRendererProvider), this viewer owns its own WebGLRenderer,
// canvas, camera, and OrbitControls so the user can orbit/zoom a single model.
//
// Ported from the reference scroll.html detail viewer (~192-218): same lights
// and material logic (reused via buildScene), auto-rotating OrbitControls with
// damping, sized to its parent container, and fully disposed on unmount.
//
// The parent is expected to wrap this in <BrowserOnly>, but we also guard with
// `typeof window` checks and only ever touch three inside a client effect, so
// the component is safe under SSR.

import React, { useEffect, useRef } from 'react';
import { buildScene, meshUrl, disposeScene } from './threeScene';

/**
 * @param {object} props
 * @param {{ file: string, kind?: string }} props.mesh - manifest entry to render.
 */
export default function ScrollViewer({ mesh }) {
  // Container that fills its parent panel; the canvas + placeholder live inside.
  const containerRef = useRef(null);

  useEffect(() => {
    if (typeof window === 'undefined') return undefined; // SSR guard
    if (!mesh || !mesh.file || !containerRef.current) return undefined;

    const container = containerRef.current;
    let cancelled = false;

    // Resources to tear down on cleanup. Captured in closure so the cleanup
    // function (returned synchronously) can reach them once the async setup
    // populates `r`.
    const r = {
      renderer: null,
      controls: null,
      scene: null,
      raf: 0,
      onResize: null,
    };

    (async () => {
      const THREE = await import('three');
      const { PLYLoader } = await import('three/examples/jsm/loaders/PLYLoader.js');
      const { OrbitControls } = await import(
        'three/examples/jsm/controls/OrbitControls.js'
      );
      if (cancelled || !containerRef.current) return;

      const W = container.clientWidth || 1;
      const H = container.clientHeight || 1;

      let renderer;
      try {
        renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
      } catch (glErr) {
        const ph = container.querySelector('.atlas-view-ph');
        if (ph) ph.textContent = '3D viewer unavailable (WebGL disabled)';
        return;
      }
      renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
      renderer.setSize(W, H);
      renderer.setClearColor(0x000000, 0);
      container.appendChild(renderer.domElement);
      r.renderer = renderer;

      r.onResize = () => {
        const w = container.clientWidth || 1;
        const h = container.clientHeight || 1;
        renderer.setSize(w, h);
        // Camera aspect is updated below once it exists.
        if (r.controls && r.controls.object) {
          r.controls.object.aspect = w / h;
          r.controls.object.updateProjectionMatrix();
        }
      };
      window.addEventListener('resize', r.onResize);

      // Load the PLY, then build the scene (reusing the shared material logic)
      // and attach interactive controls + a container-aspect camera.
      new PLYLoader().load(
        meshUrl(mesh.file),
        (geom) => {
          if (cancelled) {
            geom.dispose && geom.dispose();
            return;
          }
          const { scene, camera } = buildScene(THREE, geom, mesh);
          // Override the default camera with one sized to the container and
          // positioned per the reference detail viewer (z=3).
          camera.aspect = (container.clientWidth || 1) / (container.clientHeight || 1);
          camera.position.set(0, 0, 3);
          camera.updateProjectionMatrix();

          const controls = new OrbitControls(camera, renderer.domElement);
          controls.enableDamping = true;
          controls.autoRotate = true;
          controls.autoRotateSpeed = 1.1;

          r.scene = scene;
          r.controls = controls;

          // Remove the loading placeholder now that the model is visible.
          const ph = container.querySelector('.atlas-view-ph');
          if (ph) ph.remove();

          const loop = () => {
            r.raf = requestAnimationFrame(loop);
            controls.update();
            renderer.render(scene, camera);
          };
          r.raf = requestAnimationFrame(loop);
        },
        undefined,
        () => {
          const ph = container.querySelector('.atlas-view-ph');
          if (ph) ph.textContent = '3D load failed';
        }
      );
    })().catch(() => {
      const ph = container.querySelector('.atlas-view-ph');
      if (ph) ph.textContent = '3D viewer unavailable';
    });

    // ---- Full disposal on unmount. ----
    return () => {
      cancelled = true;
      if (r.raf) cancelAnimationFrame(r.raf);
      if (r.onResize) window.removeEventListener('resize', r.onResize);
      if (r.controls) r.controls.dispose();
      if (r.scene) disposeScene(r.scene);
      if (r.renderer) {
        // forceContextLoss() prevents leaking the WebGL context when the user
        // navigates away (browsers cap simultaneous contexts).
        r.renderer.dispose();
        r.renderer.forceContextLoss();
        if (r.renderer.domElement && r.renderer.domElement.parentNode) {
          r.renderer.domElement.parentNode.removeChild(r.renderer.domElement);
        }
      }
    };
  }, [mesh]);

  return (
    <div
      ref={containerRef}
      className="atlas-view"
      style={{ position: 'relative', width: '100%' }}
    >
      {/* Placeholder shown until the mesh loads; removed in the effect. */}
      <div
        className="atlas-view-ph"
        style={{
          position: 'absolute',
          inset: 0,
          display: 'grid',
          placeItems: 'center',
          color: '#5b6772',
          fontSize: '13px',
        }}
      >
        loading…
      </div>
    </div>
  );
}
