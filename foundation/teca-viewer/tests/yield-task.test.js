import assert from "node:assert/strict";
import { test } from "node:test";
import { yieldTask } from "../src/surface/yield-task.js";

test("a yield resolves after another task", async () => {
  let resumed = false;
  const pending = yieldTask(new AbortController().signal).then(() => {
    resumed = true;
  });
  await Promise.resolve();
  assert.equal(resumed, false);
  await pending;
  assert.equal(resumed, true);
});

test("an abort rejects a pending yield with the signal's reason", async () => {
  const controller = new AbortController();
  const reason = new Error("Work superseded");
  const aborted = yieldTask(controller.signal);
  controller.abort(reason);
  await assert.rejects(aborted, (error) => error === reason);
});
