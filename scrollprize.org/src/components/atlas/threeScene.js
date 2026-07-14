// threeScene.js
//
// Pure three.js scene-construction helpers shared by the shared-canvas grid
// renderer (AtlasRendererProvider) and the standalone interactive detail
// viewer (ScrollViewer).
//
// NOTE: This module never imports three at the top level. `THREE` is always
// passed in by the caller, which obtains it via a client-only dynamic
// `await import('three')`. This keeps the module safe to import during
// Docusaurus server-side rendering.

// Design tokens (mesh material + lights) ported from the reference renderer.
const MESH_COLOR = 0x423a32;
const MESH_ROUGHNESS = 0.62;
const MESH_METALNESS = 0.18;
const MESH_EMISSIVE = 0x0d0b09;

const HEMI_SKY = 0xfff2d8;
const HEMI_GROUND = 0x202028;
const HEMI_INTENSITY = 1.15;

const DIR_COLOR = 0xffffff;
const DIR_INTENSITY = 1.4;

const POINTS_SIZE = 0.012;
const POINTS_COLOR_NONE = 0x6a6056; // used when the cloud has no per-vertex color

/**
 * Build a scene for a single mesh/point-cloud geometry.
 *
 * Ported from the reference `buildScene` (index.html ~260-276): the geometry is
 * centered and uniformly scaled to fit a ~1.6 unit box, wrapped in a Group with
 * a slight forward tilt, and lit with a hemisphere + directional light pair.
 *
 * @param {object} THREE - the dynamically imported three module namespace.
 * @param {object} geom  - a THREE.BufferGeometry from PLYLoader.
 * @param {object} mesh  - manifest entry, e.g. { file, kind }. `kind === 'mesh'`
 *                         renders a solid Mesh; anything else renders a Points cloud.
 * @returns {{ scene: object, group: object, camera: object }}
 */
export function buildScene(THREE, geom, mesh) {
  const scene = new THREE.Scene();

  // Lights.
  scene.add(new THREE.HemisphereLight(HEMI_SKY, HEMI_GROUND, HEMI_INTENSITY));
  const dir = new THREE.DirectionalLight(DIR_COLOR, DIR_INTENSITY);
  dir.position.set(1, 1.4, 1.2);
  scene.add(dir);

  // Center + normalize scale so every model fits the same view box.
  geom.computeBoundingBox();
  const bb = geom.boundingBox;
  const center = new THREE.Vector3();
  bb.getCenter(center);
  const size = new THREE.Vector3();
  bb.getSize(size);
  const scl = 1.6 / Math.max(size.x, size.y, size.z);
  geom.translate(-center.x, -center.y, -center.z);
  geom.scale(scl, scl, scl);

  // Choose representation: solid mesh vs. point cloud.
  let obj;
  if (mesh && mesh.kind === 'mesh') {
    if (!geom.attributes.normal) geom.computeVertexNormals();
    obj = new THREE.Mesh(
      geom,
      new THREE.MeshStandardMaterial({
        color: MESH_COLOR,
        roughness: MESH_ROUGHNESS,
        metalness: MESH_METALNESS,
        emissive: MESH_EMISSIVE,
      })
    );
  } else {
    const hasColor = !!geom.attributes.color;
    obj = new THREE.Points(
      geom,
      new THREE.PointsMaterial({
        size: POINTS_SIZE,
        sizeAttenuation: true,
        vertexColors: hasColor,
        color: hasColor ? 0xffffff : POINTS_COLOR_NONE,
      })
    );
  }

  // Wrap in a group with a slight forward tilt; the group is what we spin.
  const group = new THREE.Group();
  group.add(obj);
  group.rotation.x = -0.25;
  scene.add(group);

  // Camera. Aspect starts at 1; callers update it per-frame / per-resize.
  const camera = new THREE.PerspectiveCamera(40, 1, 0.01, 100);
  camera.position.set(0, 0, 3.1);

  return { scene, group, camera };
}

/**
 * Resolve a mesh filename to its public URL under the Docusaurus static dir.
 * @param {string} file
 * @returns {string}
 */
export function meshUrl(file) {
  return `/img/data_browser/meshes/${file}`;
}

/**
 * Dispose every geometry and material reachable from a scene, freeing GPU
 * resources. Safe to call on a scene built by `buildScene`.
 * @param {object} scene - a THREE.Scene (or null/undefined, in which case it is a no-op).
 */
export function disposeScene(scene) {
  if (!scene) return;
  scene.traverse((obj) => {
    if (obj.geometry && typeof obj.geometry.dispose === 'function') {
      obj.geometry.dispose();
    }
    const mat = obj.material;
    if (!mat) return;
    if (Array.isArray(mat)) {
      mat.forEach((m) => m && typeof m.dispose === 'function' && m.dispose());
    } else if (typeof mat.dispose === 'function') {
      mat.dispose();
    }
  });
}
