/**
 * Unit tests for the two derivations the "Data & access" panel renders
 * (src/components/atlas/DataCatalog.js): the scan CONFIGURATION tuples and the
 * per-volume WEBKNOSSOS dataset list. Both are exported from buildIndex.js so
 * the build-time snapshot and the live metadata.min.json path share them.
 *
 * Run: yarn test (node --test over src/components/atlas/**\/*.test.js).
 *
 * The scan fixture is PHerc. Paris 4 (Scroll 1) exactly as the public
 * metadata.min.json states it: 7 scans, 6 distinct configurations.
 */
const test = require("node:test");
const assert = require("node:assert/strict");
const { scanConfigurations, webknossosDatasets } = require("./buildIndex");

const ESRF = "ESRF Grenoble";
const DLS = "DLS (Diamond Light Source)";

const PARIS4_SCANS = [
  { id: "20230205180739", px: 7.91, energy: 54.0, loc: DLS },
  { id: "20230206171837", px: 7.91, energy: 54.0, loc: DLS },
  { id: "20260310152857", px: 45.532, energy: 74.0, loc: ESRF },
  { id: "20260310160232", px: 45.532, energy: 110.0, loc: ESRF },
  { id: "20260311024914", px: 2.4, energy: 78.0, loc: ESRF },
  { id: "20260315072529", px: 2.4, energy: 137.0, loc: ESRF },
  { id: "20260315102010", px: 1.129, energy: 78.0, loc: ESRF },
];

test("(a) scan configurations are the tuples acquired, finest first, with scan counts", () => {
  assert.deepStrictEqual(scanConfigurations(PARIS4_SCANS), [
    { px: 1.129, energy: 78.0, loc: ESRF, n: 1 },
    { px: 2.4, energy: 78.0, loc: ESRF, n: 1 },
    { px: 2.4, energy: 137.0, loc: ESRF, n: 1 },
    { px: 7.91, energy: 54.0, loc: DLS, n: 2 },
    { px: 45.532, energy: 74.0, loc: ESRF, n: 1 },
    { px: 45.532, energy: 110.0, loc: ESRF, n: 1 },
  ]);
});

test("(b) no configuration is invented: 4 pixel sizes x 5 energies x 2 sources, 6 real tuples", () => {
  const configs = scanConfigurations(PARIS4_SCANS);
  const real = new Set(PARIS4_SCANS.map((s) => `${s.px}|${s.energy}|${s.loc}`));
  assert.strictEqual(configs.length, 6);
  assert.strictEqual(
    configs.reduce((n, c) => n + c.n, 0),
    PARIS4_SCANS.length
  );
  for (const c of configs) {
    assert.ok(real.has(`${c.px}|${c.energy}|${c.loc}`));
  }
  assert.ok(!configs.some((c) => c.px === 45.532 && c.energy === 78.0));
  assert.ok(!configs.some((c) => c.loc === DLS && c.px !== 7.91));
});

test("(c) scans missing a parameter group together instead of dropping out", () => {
  assert.deepStrictEqual(
    scanConfigurations([
      { px: 3.24, energy: null, loc: ESRF },
      { px: 3.24, energy: null, loc: ESRF },
      { px: null, energy: null, loc: null },
    ]),
    [
      { px: 3.24, energy: null, loc: ESRF, n: 2 },
      { px: null, energy: null, loc: null, n: 1 },
    ]
  );
});

test("(d) no scans yields no configurations", () => {
  assert.deepStrictEqual(scanConfigurations([]), []);
  assert.deepStrictEqual(scanConfigurations(undefined), []);
});

test("(e) every curated WEBKNOSSOS dataset is offered, in curated order", () => {
  const progress = {
    wk: [
      { name: "PHerc0343P-4um", px: 8.64, url: "https://wk.aws.ash2txt.org/datasets/6867f0ff010000e305d2f8c2/view" },
      { name: "PHerc0343P-2um", px: 2.215, url: "https://wk.aws.ash2txt.org/datasets/6867f0de0100000806d2f8bf/view" },
    ],
  };
  assert.deepStrictEqual(webknossosDatasets(progress), progress.wk);
});

test("(f) the legacy single wkUrl still renders, as a one-dataset list", () => {
  const url = "https://wk.aws.ash2txt.org/datasets/67a53a6c01000001019dbc46/view";
  assert.deepStrictEqual(webknossosDatasets({ wkUrl: url }), [
    { name: null, px: null, url },
  ]);
});

test("(g) a wk list wins over a stale wkUrl, and entries without a url are dropped", () => {
  const url = "https://wk.aws.ash2txt.org/datasets/68936bbf01000002015bf075/view";
  assert.deepStrictEqual(
    webknossosDatasets({ wkUrl: "https://wk.aws.ash2txt.org/datasets/old/view", wk: [{ name: "PHerc0139", px: 9.362, url }, { name: "no link" }] }),
    [{ name: "PHerc0139", px: 9.362, url }]
  );
});

test("(h) a scroll with no WEBKNOSSOS data offers no button", () => {
  assert.deepStrictEqual(webknossosDatasets({ wkUrl: null }), []);
  assert.deepStrictEqual(webknossosDatasets({ wk: [] }), []);
  assert.deepStrictEqual(webknossosDatasets(null), []);
});
