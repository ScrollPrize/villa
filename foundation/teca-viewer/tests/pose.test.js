import assert from "node:assert/strict";
import { test } from "node:test";
import { IDENTITY_ROTATION, quatFromAxisAngle } from "../src/math.js";
import { createViewerSession } from "../src/session.js";
import { drawnPanes, drawnPlanes, panelVoxelOffset, shownView, viewIsRotated } from "../src/views/pose.js";

test("in pane mode only the tilted pane takes the rotated path", () => {
  const state = createViewerSession();
  state.rotationMode = "pane";
  state.paneRotations = { xy: IDENTITY_ROTATION, zy: quatFromAxisAngle([0, 1, 0], 12), xz: IDENTITY_ROTATION };
  assert.deepEqual(
    ["xy", "zy", "xz"].map((plane) => viewIsRotated(state, plane)),
    [false, true, false],
  );
});

test("a maximized pane is the only one drawn, but the 3D pane's three planes are drawn with it", () => {
  const state = createViewerSession();
  assert.deepEqual(drawnPlanes(state), ["xy", "zy", "xz"]);
  state.fullscreenViewport = "xy";
  assert.deepEqual(drawnPlanes(state), ["xy"]);
  assert.deepEqual(drawnPlanes(state, ["zy", "xy"]), ["xy"]);
  assert.deepEqual(drawnPanes(state, ["xy", "zy", "xz", "3d"]), ["xy"]);
  state.fullscreenViewport = "3d";
  assert.deepEqual(drawnPlanes(state), ["xy", "zy", "xz"]);
  assert.deepEqual(drawnPanes(state, ["xy", "zy", "xz", "3d", "surface"]), ["xy", "zy", "xz", "3d"]);
  assert.deepEqual(drawnPanes(state, ["3d"]), ["3d"]);
});

test("a docked Surface draws its planes beside it, or only the pane maximized in the dock, and all three with 3D", () => {
  const state = createViewerSession();
  state.fullscreenViewport = "surface";
  assert.deepEqual(drawnPlanes(state), []);
  for (const mode of ["below", "right"]) {
    state.surfaceView.layout = { mode, planes: ["xy", "xz"], split: 0.63, weights: [1.2, 1, 0.8, 1], maximized: null };
    assert.deepEqual(drawnPlanes(state), ["xy", "xz"]);
    assert.deepEqual(drawnPanes(state, ["xy", "zy", "xz", "3d", "surface"]), ["xy", "xz", "surface"]);
    state.surfaceView.layout.maximized = "xz";
    assert.deepEqual(drawnPanes(state, ["xy", "zy", "xz", "surface"]), ["xz"]);
    state.surfaceView.layout.maximized = "surface";
    assert.deepEqual(drawnPlanes(state), []);
    state.surfaceView.layout = { mode, planes: ["3d"], split: 0.63, weights: [1, 1, 1, 1], maximized: null };
    assert.deepEqual(drawnPanes(state, ["xy", "zy", "xz", "3d", "surface"]), ["xy", "zy", "xz", "3d", "surface"]);
    state.surfaceView.layout.maximized = "3d";
    assert.deepEqual(drawnPanes(state, ["xy", "zy", "xz", "3d", "surface"]), ["xy", "zy", "xz", "3d"]);
    state.surfaceView.layout.maximized = "surface";
    assert.deepEqual(drawnPanes(state, ["xy", "zy", "xz", "3d", "surface"]), ["surface"]);
  }
});

test("the view shown is the grid, a pane alone, or the Surface unless a pane is maximized in its dock", () => {
  const state = createViewerSession();
  assert.equal(shownView(state), "grid");
  state.fullscreenViewport = "3d";
  assert.equal(shownView(state), "3d");
  state.fullscreenViewport = "surface";
  state.surfaceView.layout = { mode: "solo", planes: ["xy"], split: 0.7, weights: [1, 1, 1, 1], maximized: "xy" };
  assert.equal(shownView(state), "surface");
  state.surfaceView.layout.mode = "below";
  assert.equal(shownView(state), "xy");
  state.surfaceView.layout.maximized = null;
  assert.equal(shownView(state), "surface");
});

test("the pane center is offset 0, and the top middle of a 2:1 pane 400 voxels high is 200 voxels up", () => {
  assert.deepEqual(panelVoxelOffset(50, 50, 0, 0, 100, 100, 2, 400), { u: 0, v: 0 });
  assert.deepEqual(panelVoxelOffset(100, 0, 0, 0, 200, 100, 1, 400), { u: 0, v: -200 });
});
