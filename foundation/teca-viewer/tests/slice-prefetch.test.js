import assert from "node:assert/strict";
import { test } from "node:test";
import { SlicePrefetch } from "../src/render/slice-prefetch.js";
import { createViewerSession } from "../src/session.js";
import { createViewerState } from "../src/state.js";

const SCALE = {
  shape_ZYX: [1024, 512, 512],
  chunks_ZYX: [128, 128, 128],
  downsample_ZYX: [1, 1, 1],
  offset_ZYX: [0, 0, 0],
};

test("maximised 3D pane: the step plane XY reads its neighbours along z, nearer first, and ZY its own alone", () => {
  const state = createViewerState(createViewerSession());
  state.fullscreenViewport = "3d";
  state.activeViewport = "3d";
  state.center_ZYX = [300, 100, 100];
  const prefetch = new SlicePrefetch({ state, view: null });
  const addresses = [{ chunk_ZYX: [2, 0, 0], fetchOrder: 0 }];
  const fetched = (plane) =>
    prefetch.planeFetchAddresses({ plane }, SCALE, addresses, []).map((a) => [...a.chunk_ZYX, a.fetchOrder]);
  assert.deepEqual(fetched("xy"), [
    [2, 0, 0, 0],
    [1, 0, 0, 1],
    [3, 0, 0, 1],
  ]);
  assert.deepEqual(fetched("zy"), [[2, 0, 0, 0]]);
});
