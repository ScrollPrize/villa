import * as THREE from "../../vendor/three.module.js";
import { HIDDEN_PANE_MAX_PIXELS } from "../config.js";
import { elementRect } from "../element-rect.js";
import { PaneFramebuffer } from "../gpu/pane-framebuffer.js";

const VIEW_WIDTH_UNITS = 1.15;
const CAMERA_DISTANCE = 3.2;
const CAMERA_FOV_DEGREES = 45;

// Face-on, VIEW_WIDTH_UNITS spans 0.434 of the pane's height. A tilted plane covers 0.47 to 0.71 of that at the opening
// orbit, its corner nearest the camera at most 1.34 pixels a texel, within SCALE_PICK_TOLERANCE's 1.35. A picture that
// much wider would be 1.8 times the pixels, its texels dropped, sampled nearest, where it is face-on.
export function planeSpanPixels(height, pixelRatio) {
  const viewHeightUnits = 2 * CAMERA_DISTANCE * Math.tan((CAMERA_FOV_DEGREES * Math.PI) / 360);
  return Math.ceil((Math.floor(height * pixelRatio) * VIEW_WIDTH_UNITS) / viewHeightUnits);
}

export class ThreePane {
  constructor(container, orthogonalPanes, pixelRatio, rendering) {
    this.name = "3d";
    this.mode = "3d";
    this.element = document.createElement("div");
    this.element.className = "viewer-pane pane-3d";
    this.element.dataset.plane = this.name;
    this.canvas = document.createElement("canvas");
    this.canvas.id = "viewer-3d";
    this.canvas.tabIndex = 0;
    this.canvas.setAttribute("aria-label", "Interactive 3D overview of the slice planes");
    const paneName = document.createElement("div");
    paneName.className = "pane-name";
    paneName.textContent = "3D";
    const maximize = document.createElement("button");
    maximize.type = "button";
    maximize.className = "pane-maximize";
    maximize.dataset.plane = this.name;
    maximize.textContent = "⛶";
    maximize.setAttribute("aria-label", `Maximize ${this.name.toUpperCase()} pane`);
    maximize.title = "Maximize pane (Z)";
    const corner = document.createElement("div");
    corner.className = "pane-corner-controls";
    corner.append(maximize);
    this.element.append(corner);
    this.surfaceButton = document.createElement("button");
    this.surfaceButton.type = "button";
    this.surfaceButton.className = "pane-return";
    this.surfaceButton.textContent = "Surface…";
    this.surfaceButton.title = "Open a tifxyz Surface (U).";
    this.element.append(this.surfaceButton);
    this.scaleBar = document.createElement("div");
    this.scaleBar.className = "scale-bar";
    this.element.append(this.canvas, paneName, this.scaleBar);
    container.append(this.element);
    this.orthogonalPanes = [...orthogonalPanes];
    this.pixelRatio = pixelRatio;
    this.renderer = rendering.renderer;
    this.output = new PaneFramebuffer(rendering, this.canvas);
    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(CAMERA_FOV_DEGREES, 1, 0.1, 100);
    this.scene.add(new THREE.AmbientLight(0xffffff, 1));
    this.world = new THREE.Group();
    this.world.matrixAutoUpdate = false;
    this.scene.add(this.world);
    this.fallbackScene = new THREE.Scene();
    this.fallbackCamera = new THREE.PerspectiveCamera(CAMERA_FOV_DEGREES, 1, 0.1, 100);
    this.tileGeometry = new THREE.PlaneGeometry(1, 1);
    this.fallbackOverview = this.createFallbackOverview();
    this.fallbackScene.add(this.fallbackOverview);
    this.imageEntries = [];
    this.imageOverview = null;
    this.renderedFrame = null;
    this.raycaster = new THREE.Raycaster();
  }

  panel() {
    return [
      this.name,
      0,
      0,
      this.canvas.width || Math.max(1, Math.round(this.canvas.clientWidth * this.pixelRatio)),
      this.canvas.height || Math.max(1, Math.round(this.canvas.clientHeight * this.pixelRatio)),
    ];
  }

  setVisible(visible) {
    this.element.hidden = !visible;
    if (!visible) this.releaseSourceResources();
  }

  contains(clientX, clientY) {
    const rect = elementRect(this.canvas);
    return clientX >= rect.left && clientX <= rect.right && clientY >= rect.top && clientY <= rect.bottom;
  }

  resize() {
    this.output.resize();
  }

  hiddenPlaneSide() {
    const height = this.canvas.clientHeight;
    if (height === 0) return null;
    return Math.min(planeSpanPixels(height, this.pixelRatio), HIDDEN_PANE_MAX_PIXELS);
  }

  render(_orthogonalScene, cameraOrientation, overlayContext) {
    this.output.begin(false);
    const sourceActive = overlayContext.sourceActive;
    if (!sourceActive) this.releaseSourceResources();
    else if (this.imageOverview === null) {
      this.imageOverview = this.createImageOverview();
      this.scene.add(this.imageOverview);
    }
    const [, , , panelWidth, panelHeight] = this.panel();
    this.renderer.setClearColor(0x000000, 1);
    this.renderer.clear(true, true, true);
    const [scene, camera, subject] =
      sourceActive && this.updateImageOverview(overlayContext.center_ZYX, overlayContext.viewWidthVoxels)
        ? [this.scene, this.camera, this.imageOverview]
        : [this.fallbackScene, this.fallbackCamera, this.fallbackOverview];
    subject.visible = true;
    subject.quaternion.identity();
    const orbit = new THREE.Quaternion(...cameraOrientation);
    camera.aspect = panelWidth / panelHeight;
    camera.zoom = 1;
    camera.near = 0.1;
    camera.far = 100;
    camera.up.set(0, 1, 0).applyQuaternion(orbit);
    camera.position.set(0, 0, CAMERA_DISTANCE).applyQuaternion(orbit);
    camera.lookAt(0, 0, 0);
    camera.updateProjectionMatrix();
    this.renderer.setViewport(0, 0, panelWidth / this.pixelRatio, panelHeight / this.pixelRatio);
    this.renderer.setScissor(0, 0, panelWidth / this.pixelRatio, panelHeight / this.pixelRatio);
    this.renderer.render(scene, camera);
    subject.visible = false;
    this.renderedFrame =
      scene === this.scene
        ? { center_ZYX: [...overlayContext.center_ZYX], width: overlayContext.viewWidthVoxels }
        : null;
    this.output.complete();
  }

  addObject(object) {
    this.world.add(object);
  }

  removeObject(object) {
    this.world.remove(object);
  }

  pointAtClient(clientX, clientY) {
    if (this.renderedFrame === null || this.element.hidden) return null;
    const rect = elementRect(this.canvas);
    if (!this.contains(clientX, clientY) || rect.width === 0 || rect.height === 0) return null;
    this.raycaster.setFromCamera(
      new THREE.Vector2((2 * (clientX - rect.left)) / rect.width - 1, 1 - (2 * (clientY - rect.top)) / rect.height),
      this.camera,
    );
    const quads = this.imageEntries.filter((entry) => entry.mesh.visible).map((entry) => entry.mesh);
    const hit = this.raycaster.intersectObjects([...quads, ...this.world.children], true)[0];
    if (hit === undefined) return null;
    if (!quads.includes(hit.object)) {
      const world_XYZ = this.world.worldToLocal(hit.point.clone());
      return [world_XYZ.z, world_XYZ.y, world_XYZ.x];
    }
    const point_XYZ = this.imageOverview.worldToLocal(hit.point.clone());
    return [point_XYZ.z, point_XYZ.y, point_XYZ.x].map(
      (value, axis) => this.renderedFrame.center_ZYX[axis] + value * this.renderedFrame.width,
    );
  }

  releaseSourceResources() {
    this.renderedFrame = null;
    if (this.imageOverview === null) return;
    for (const entry of this.imageEntries) entry.mesh.material.dispose();
    this.imageEntries = [];
    this.imageOverview.removeFromParent();
    this.imageOverview = null;
  }

  createFallbackOverview() {
    const group = new THREE.Group();
    group.visible = false;
    group.scale.setScalar(0.85);
    const slicePlanes = [
      { color: 0xe67e80, rotation: [0, 0, 0], position: [0, 0, 0.001] },
      { color: 0x7fbbb3, rotation: [0, Math.PI / 2, 0], position: [0.001, 0, 0] },
      { color: 0xa7c080, rotation: [Math.PI / 2, 0, 0], position: [0, 0.001, 0] },
    ];
    for (const plane of slicePlanes) {
      const mesh = new THREE.Mesh(
        this.tileGeometry,
        new THREE.MeshBasicMaterial({
          color: plane.color,
          side: THREE.DoubleSide,
          transparent: true,
          opacity: 0.3,
          depthWrite: false,
        }),
      );
      mesh.position.set(...plane.position);
      mesh.rotation.set(...plane.rotation);
      mesh.renderOrder = -1;
      group.add(mesh);
    }
    group.add(
      new THREE.LineSegments(
        new THREE.EdgesGeometry(new THREE.BoxGeometry(1, 1, 1)),
        new THREE.LineBasicMaterial({ color: 0x9aa0a8, transparent: true, opacity: 0.72 }),
      ),
    );
    const axes_XYZ = [
      [[-0.5, 0, 0], [0.5, 0, 0], 0xe67e80],
      [[0, -0.5, 0], [0, 0.5, 0], 0xa7c080],
      [[0, 0, -0.5], [0, 0, 0.5], 0x7fbbb3],
    ];
    for (const [start, end, color] of axes_XYZ) {
      const geometry = new THREE.BufferGeometry().setFromPoints([
        new THREE.Vector3(...start),
        new THREE.Vector3(...end),
      ]);
      group.add(new THREE.Line(geometry, new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.9 })));
    }
    group.add(
      new THREE.LineSegments(
        new THREE.EdgesGeometry(new THREE.BoxGeometry(0.04, 0.04, 0.04)),
        new THREE.LineBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.8 }),
      ),
    );
    return group;
  }

  createImageOverview() {
    const group = new THREE.Group();
    group.visible = false;
    group.scale.setScalar(VIEW_WIDTH_UNITS);
    for (const pane of this.orthogonalPanes) {
      const mesh = new THREE.Mesh(
        this.tileGeometry,
        new THREE.MeshBasicMaterial({
          map: pane.output.target.texture,
          side: THREE.DoubleSide,
          depthTest: true,
          depthWrite: true,
        }),
      );
      mesh.matrixAutoUpdate = false;
      this.imageEntries.push({ pane, mesh });
      group.add(mesh);
    }
    return group;
  }

  updateImageOverview(center_ZYX, viewWidthVoxels) {
    const unit = VIEW_WIDTH_UNITS / viewWidthVoxels;
    this.world.matrix.makeScale(unit, unit, unit).setPosition(...center_ZYX.toReversed().map((value) => -value * unit));
    this.world.matrixWorldNeedsUpdate = true;
    let ready = false;
    for (const { pane, mesh } of this.imageEntries) {
      const canvas = pane.canvas;
      mesh.visible = pane.rasterPose !== null && canvas.width > 1 && canvas.height > 1;
      if (!mesh.visible) continue;
      const aspect = canvas.width / canvas.height;
      const { basis, center_ZYX: rasterCenter_ZYX, viewWidthVoxels: rasterWidth } = pane.rasterPose;
      const scale = rasterWidth / viewWidthVoxels;
      mesh.matrix.makeBasis(
        new THREE.Vector3(...basis.right_XYZ).multiplyScalar(scale * Math.max(1, aspect)),
        new THREE.Vector3(...basis.up_XYZ).multiplyScalar(scale * Math.max(1, 1 / aspect)),
        new THREE.Vector3(...basis.normal_XYZ),
      );
      mesh.matrix.setPosition(
        ...[2, 1, 0].map((axis) => (rasterCenter_ZYX[axis] - center_ZYX[axis]) / viewWidthVoxels),
      );
      mesh.matrixWorldNeedsUpdate = true;
      ready = true;
    }
    return ready;
  }
}
