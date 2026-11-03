import assert from "node:assert/strict";
import { test } from "node:test";
import { formatBytes } from "../src/ui/format-bytes.js";

test("1023 bytes, 1536 bytes, 5 MiB and 3 GiB read 1023 B, 1.5 KiB, 5.0 MiB and 3.00 GiB", () => {
  assert.deepEqual([1023, 1536, 5 * 2 ** 20, 3 * 2 ** 30].map(formatBytes), [
    "1023 B",
    "1.5 KiB",
    "5.0 MiB",
    "3.00 GiB",
  ]);
});
