import assert from "node:assert/strict";
import { test } from "node:test";
import { IDENTITY_ROTATION, quatFromAxisAngle } from "../src/math.js";
import { rotationReadout } from "../src/ui/rotation-readout.js";

const TILTED = quatFromAxisAngle([1, 0, 0], 3.2);
const ALIGNED = { xy: IDENTITY_ROTATION, zy: IDENTITY_ROTATION, xz: IDENTITY_ROTATION };

test("a frame rotation of 3.2 degrees reads as all panes tilted 3.2°", () => {
  assert.equal(
    rotationReadout({ plane: "zy", rotationMode: "frame", rotation: TILTED, paneRotations: ALIGNED }),
    "All panes tilted 3.2°",
  );
});

test("an aligned active pane with a tilted XZ pane reads both", () => {
  assert.equal(
    rotationReadout({
      plane: "zy",
      rotationMode: "pane",
      rotation: IDENTITY_ROTATION,
      paneRotations: { ...ALIGNED, xz: TILTED },
    }),
    "Active ZY axis-aligned, XZ tilted 3.2°",
  );
});
