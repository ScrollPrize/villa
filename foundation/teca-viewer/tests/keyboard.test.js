import assert from "node:assert/strict";
import { test } from "node:test";
import { pressedBinding } from "../src/ui/keyboard.js";

const TOGGLE = { code: "KeyV", shift: false, run: () => {} };
const STEP = { code: "BracketRight", shift: false, repeats: true, run: () => {} };
const TOOL_TOGGLE = { code: "KeyW", shift: false, tool: "pen", run: () => {} };

function keydown(code, repeat) {
  return { code, key: "", repeat, shiftKey: false, ctrlKey: false, metaKey: false, altKey: false };
}

test("a key held runs its binding once, unless the binding repeats, and its repeats stay the binding's", () => {
  const bindings = [TOGGLE, STEP, TOOL_TOGGLE];
  const pressed = [
    ["KeyV", false, null],
    ["KeyV", true, null],
    ["BracketRight", false, null],
    ["BracketRight", true, null],
    ["KeyW", false, "pen"],
    ["KeyW", true, "pen"],
    ["KeyW", false, null],
  ].map(([code, repeat, tool]) => {
    const { binding, runs } = pressedBinding(bindings, keydown(code, repeat), tool);
    return [binding?.code, runs];
  });
  assert.deepEqual(pressed, [
    ["KeyV", true],
    ["KeyV", false],
    ["BracketRight", true],
    ["BracketRight", true],
    ["KeyW", true],
    ["KeyW", false],
    [undefined, false],
  ]);
});
