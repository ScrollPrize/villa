import * as THREE from "../../vendor/three.module.js";
import { dot_ZYX } from "../math.js";
import { SurfaceIntersections } from "../surface/intersections.js";
import { paneBasis, surfaceIntersectionSegments } from "../surface/overlay-geometry.js";
import { overlayColor } from "../surface/overlay-store.js";
import { drawnPlanes, planeDisplayBasis, planeRotation, viewWidth } from "./pose.js";

const PANES = ["xy", "zy", "xz"];
const LINE_OPACITY = 0.9;
const LINE_RENDER_ORDER = 2000;
const EMPTY_OVERLAY_CONTEXT = Object.freeze({ sourceActive: false });

function lineGeometry(segments_uv, colors) {
  const geometry = new THREE.InstancedBufferGeometry();
  geometry.setAttribute(
    "position",
    new THREE.Float32BufferAttribute([0, 1, 0, 0, -1, 0, 1, -1, 0, 0, 1, 0, 1, -1, 0, 1, 1, 0], 3),
  );
  geometry.setAttribute("segmentUv", new THREE.InstancedBufferAttribute(new Float32Array(segments_uv), 4));
  geometry.setAttribute("segmentColor", new THREE.InstancedBufferAttribute(new Float32Array(colors), 3));
  geometry.instanceCount = segments_uv.length / 4;
  return geometry;
}

// LineBasicMaterial ignores linewidth on WebGL, so segments are drawn as quads with a screen-constant width.
function lineMaterial() {
  return new THREE.ShaderMaterial({
    uniforms: { halfWidthUv: { value: 0 }, opacity: { value: LINE_OPACITY } },
    vertexShader: `
      attribute vec4 segmentUv;
      attribute vec3 segmentColor;
      uniform float halfWidthUv;
      varying vec3 vColor;
      void main() {
        vec2 direction = segmentUv.zw - segmentUv.xy;
        float directionLength = length(direction);
        vec2 perpendicular = directionLength > 0.0 ? vec2(-direction.y, direction.x) / directionLength : vec2(0.0);
        vec2 uv = mix(segmentUv.xy, segmentUv.zw, position.x) + perpendicular * position.y * halfWidthUv;
        vColor = segmentColor;
        gl_Position = projectionMatrix * modelViewMatrix * vec4(uv, 0.0, 1.0);
      }
    `,
    fragmentShader: `
      uniform float opacity;
      varying vec3 vColor;
      void main() {
        gl_FragColor = vec4(vColor, opacity);
      }
    `,
    transparent: true,
    depthTest: false,
    depthWrite: false,
  });
}

export function createOverlayRenderer({ state, view, overlays, overlayRevision }) {
  const paneObjects = new Map();
  const surfaceIntersections = new SurfaceIntersections({
    isInteracting: () =>
      state.pointerDrag !== null ||
      state.heldSliceDirection !== 0 ||
      state.heldZoomDirection !== 0 ||
      state.heldRotation !== null,
  });

  function paneObjectsFor(pane) {
    if (paneObjects.has(pane)) return paneObjects.get(pane);
    const group = new THREE.Group();
    const lines = new THREE.Mesh(new THREE.InstancedBufferGeometry(), lineMaterial());
    lines.renderOrder = LINE_RENDER_ORDER;
    lines.frustumCulled = false;
    lines.visible = false;
    group.add(lines);
    view.scene.tileGroups.get(pane).add(group);
    const detailStatus = document.createElement("div");
    detailStatus.className = "pane-status";
    detailStatus.hidden = true;
    detailStatus.setAttribute("role", "status");
    view.getCanvas(pane).parentElement.append(detailStatus);
    const objects = { group, lines, detailStatus, refinements: new Map(), signature: null, overlayStats: [] };
    paneObjects.set(pane, objects);
    return objects;
  }

  function disposePaneObjects() {
    surfaceIntersections.retain(new Set());
    for (const objects of paneObjects.values()) {
      objects.group.removeFromParent();
      objects.detailStatus.remove();
      objects.lines.geometry.dispose();
      objects.lines.material.dispose();
    }
    paneObjects.clear();
  }

  function updateOverlays(planes, requestRedraw) {
    if (state.volume === null) {
      disposePaneObjects();
      return EMPTY_OVERLAY_CONTEXT;
    }
    const visible = overlays.filter((overlay) => overlay.visible);
    const refinementKeys = new Set();
    const drawn = drawnPlanes(state);
    for (const [pane, objects] of paneObjects) {
      for (const [overlay, key] of objects.refinements) {
        if (!drawn.includes(pane) || !visible.includes(overlay)) objects.refinements.delete(overlay);
        else refinementKeys.add(key);
      }
    }
    const span = viewWidth(state);
    for (const pane of planes) {
      const rotation = planeRotation(state, pane);
      const basis = paneBasis(pane, rotation);
      const iso = dot_ZYX(state.center_ZYX, basis.normal_ZYX);
      const objects = paneObjectsFor(pane);
      const pixelsPerVoxel = view.getOrthogonalPanelPixelSize(pane) / span;
      const viewAspect = view.getOrthogonalPanelAspect(pane);
      objects.lines.material.uniforms.halfWidthUv.value = state.overlayLineWidth / 2 / Math.max(pixelsPerVoxel, 1e-6);
      const refinements = new Map();
      const errors = [];
      for (const overlay of visible) {
        let key = objects.refinements.get(overlay);
        if (key === undefined) {
          key = {};
          objects.refinements.set(overlay, key);
        }
        refinementKeys.add(key);
        const planeSignature = `${basis.normal_ZYX}:${basis.right_ZYX}:${basis.up_ZYX}:${overlay.registration.scale_ZYX}:${overlay.registration.offset_ZYX}`;
        const refinement = surfaceIntersections.request(
          key,
          {
            image: overlay.positionImage,
            pyramid: overlay.pyramid,
            registration: overlay.registration,
            planeSignature,
            signature: `${planeSignature}:${state.center_ZYX.join(",")}:${span}:${viewAspect}:${pixelsPerVoxel}`,
            pane: {
              basis,
              center_ZYX: [...state.center_ZYX],
              voxelsPerPixel: 1 / pixelsPerVoxel,
              halfSpan_UV: [(span * Math.max(1, viewAspect)) / 2, (span * Math.max(1, 1 / viewAspect)) / 2],
            },
          },
          requestRedraw,
        );
        refinements.set(overlay, refinement);
        if (refinement.error !== null) errors.push(`Mesh detail: ${refinement.error}`);
      }
      objects.detailStatus.textContent = errors.join("; ");
      objects.detailStatus.hidden = errors.length === 0;
      const cutSignature =
        visible.length === 0 ? "" : `:${rotation.map((value) => value.toFixed(5)).join(",")}:${iso.toFixed(5)}`;
      const signature = `${overlayRevision()}${cutSignature}:${[...refinements.values()].map((entry) => entry.version).join(",")}`;
      if (objects.signature !== signature) {
        objects.signature = signature;
        const segments_uv = [];
        const colors = [];
        objects.overlayStats = visible.map((overlay) => {
          const refined = refinements.get(overlay).result;
          const evidence = refined?.evidence ?? {};
          const segments = surfaceIntersectionSegments(
            overlay.points_zyx_HW3,
            overlay.grid_HW[0],
            overlay.grid_HW[1],
            basis,
            iso,
            overlay.surfaceIntersectionIndex,
            refined ? {} : evidence,
            refined ? { ...refined.geometry.coverage, stride: overlay.sourceStride } : null,
          );
          if (refined) for (const value of refined.segments_uv) segments.push(value);
          const color = new THREE.Color(overlayColor(overlay));
          const start = segments_uv.length;
          for (let i = 0; i < segments.length; i += 4) {
            if (segments[i] === segments[i + 2] && segments[i + 1] === segments[i + 3]) continue;
            segments_uv.push(segments[i], segments[i + 1], segments[i + 2], segments[i + 3]);
            colors.push(color.r, color.g, color.b);
          }
          return {
            path: overlay.path,
            lineVertices: ((segments_uv.length - start) / 4) * 6,
            surfaceIntersection: evidence,
          };
        });
        objects.lines.geometry.dispose();
        objects.lines.geometry = lineGeometry(segments_uv, colors);
        objects.lines.visible = segments_uv.length > 0;
      }
      const k = 2 / span;
      objects.group.position.set(
        -dot_ZYX(state.center_ZYX, basis.right_ZYX) * k,
        -dot_ZYX(state.center_ZYX, basis.up_ZYX) * k,
        0,
      );
      objects.group.scale.set(k, k, 1);
    }
    surfaceIntersections.retain(refinementKeys);
    return {
      sourceActive: true,
      planeBases: Object.fromEntries(PANES.map((plane) => [plane, planeDisplayBasis(state, plane)])),
      center_ZYX: state.center_ZYX,
      viewWidthVoxels: span,
    };
  }

  function getOverlay2dStats() {
    return Object.fromEntries(
      PANES.map((pane) => {
        const objects = paneObjects.get(pane);
        const overlayStats = structuredClone(objects?.overlayStats ?? []);
        const lineVertices = overlayStats.reduce((total, overlay) => total + overlay.lineVertices, 0);
        const cutting = objects === undefined ? 0 : surfaceIntersections.cutting(objects.refinements.values());
        return [pane, { lineVertices, cutting, overlays: overlayStats }];
      }),
    );
  }

  return { updateOverlays, getOverlay2dStats };
}
