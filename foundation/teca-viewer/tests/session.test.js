import assert from "node:assert/strict";
import { Buffer } from "node:buffer";
import { test } from "node:test";
import { SESSION_JSON_BYTE_LIMIT } from "../src/config.js";
import {
  createViewerSession,
  deserializeSession,
  linkedSession,
  readSessionFile,
  sessionSnapshot,
  sessionToken,
} from "../src/session.js";

const token = (session) => Buffer.from(JSON.stringify(session), "utf8").toString("base64url");
const withPatch = (patch) => ({ ...sessionSnapshot(createViewerSession()), ...patch });

test("a session with Unicode sources, an overlay and a Surface dock round-trips through its token", () => {
  const session = createViewerSession();
  const papiro = "https://example.test/dados/papiro-α.ome.zarr/";
  session.sources = [papiro, "https://example.test/scan.zarr/"];
  session.activeSource = papiro;
  session.center_ZYX = [7, 11, 13];
  session.rotation = [0, Math.SQRT1_2, 0, Math.SQRT1_2];
  session.cameraOrientation = [Math.SQRT1_2, 0, 0, Math.SQRT1_2];
  session.heldSliceStep = 7;
  session.manualScaleIndex = 2;
  session.manualCoarsestScaleIndex = 2;
  session.overlays = [
    { path: "https://example.test/folha-ç.tifxyz/", primarySource: papiro, visible: true, colorIndex: 0 },
  ];
  session.fullscreenViewport = "surface";
  session.activeViewport = "xz";
  session.surfaceView.path = session.overlays[0].path;
  session.surfaceView.primarySource = papiro;
  session.surfaceView.layout = {
    mode: "right",
    planes: ["xy", "xz"],
    split: 0.63,
    weights: [1.2, 1, 0.8, 1],
    maximized: null,
  };
  session.tabs = { left: "view", right: "mesh-edit" };
  const snapshot = sessionSnapshot(session);
  assert.deepEqual(deserializeSession(token(snapshot)), snapshot);
});

test("a malformed session names what is wrong with it", () => {
  for (const [patch, message] of [
    [{ version: 1 }, /^Session keys must be exactly sources, activeSource,/],
    [{ rotationMode: "euler" }, "Unknown rotation mode: euler."],
    [{ extra: true }, /^Session keys must be exactly sources, activeSource,/],
    [{ activeSource: "missing" }, "Active source missing is absent from session sources."],
    [{ sources: [{ id: "ct", location: "https://example.test/ct.zarr/" }] }, /is an HTTP\(S\) URL/],
    [{ rotation: [1, 0, 0, 1] }, /rotation must be a unit quaternion/],
    [{ cameraOrientation: [0, 0, 0, 0] }, /cameraOrientation must be a unit quaternion/],
    [{ heldSliceStep: 0 }, "Session heldSliceStep must be an integer inside 1..20."],
    [{ overlayLineWidth: 31 }, "Session overlayLineWidth 31 is outside 0.5..30."],
    [{ manualScaleIndex: 3, manualCoarsestScaleIndex: 2 }, /finest level must not be coarser/],
    [
      { tabs: { left: "info", right: null } },
      `Session tabs must be the left's data or view and the right's name or null: {"left":"info","right":null}.`,
    ],
    [{ tabs: { left: "data", right: "Points" } }, /^Session tabs must be the left's data or view/],
    [{ tabs: { left: "data" } }, /^Session tabs must be the left's data or view/],
  ]) {
    assert.throws(() => deserializeSession(token(withPatch(patch))), { message }, JSON.stringify(patch));
  }
});

test("each extension's entry round-trips, and an extension named in camel case is refused", () => {
  const session = createViewerSession();
  session.extensions = { layers: [{ location: "https://example.test/ink.zarr/", color: "#ff00ff" }] };
  const snapshot = sessionSnapshot(session);
  assert.deepEqual(deserializeSession(token(snapshot)), snapshot);
  assert.throws(() => deserializeSession(token(withPatch({ extensions: { inkLayers: [] } }))), {
    message: 'Session extensions must be an object keyed by kebab-case names: {"inkLayers":[]}.',
  });
});

test("a session's token holds its JSON up to SESSION_JSON_BYTE_LIMIT bytes, and a byte past it is refused", () => {
  const session = createViewerSession();
  session.extensions = { filler: "" };
  const unfilled = Buffer.from(sessionToken(session).token, "base64url").byteLength;
  session.extensions.filler = "x".repeat(SESSION_JSON_BYTE_LIMIT - unfilled);
  const full = sessionToken(session);
  assert.equal(Buffer.from(full.token, "base64url").byteLength, 32768);
  assert.deepEqual([deserializeSession(full.token), full.refused], [sessionSnapshot(session), null]);
  session.extensions.filler += "x";
  const past = sessionToken(session);
  assert.equal(past.token, null);
  assert.match(
    past.refused,
    /^The link holds the view as it last fitted: this one is 32,769 bytes, over 32,768, and filler /,
  );
});

test("a session past the limit is refused naming its largest part, a field or an extension's entry", () => {
  const session = createViewerSession();
  session.extensions = { small: "y".repeat(1000), large: "x".repeat(40000) };
  assert.match(
    sessionToken(session).refused,
    /this one is [\d,]+ bytes, over 32,768, and large take 40,002 of them\.$/,
  );
  session.extensions = { small: "y".repeat(1000) };
  session.sources = [`https://example.test/${"a".repeat(40000)}/`];
  assert.match(
    sessionToken(session).refused,
    /this one is [\d,]+ bytes, over 32,768, and sources take 40,026 of them\.$/,
  );
});

test("the address's #session= is the session it links to, and a malformed one is refused", () => {
  const snapshot = withPatch({ center_ZYX: [7, 11, 13] });
  const defaults = createViewerSession();
  for (const [address, linked] of [
    [
      { search: "", hash: "" },
      { session: defaults, refused: null },
    ],
    [
      { search: "?volume=https://example.test/ct.zarr/", hash: `#session=${token(snapshot)}` },
      { session: snapshot, refused: null },
    ],
    [
      { search: "", hash: `#session=${token(withPatch({ rotationMode: "euler" }))}` },
      { session: defaults, refused: "Session link rejected: Unknown rotation mode: euler." },
    ],
  ])
    assert.deepEqual(linkedSession(address), linked, JSON.stringify(address).slice(0, 80));
});

test("a session file opens as its session's link with what it kept, and one that is not, or past the limit, is refused", () => {
  const session = withPatch({ center_ZYX: [7, 11, 13] });
  const kept = { points: { ct: "https://example.test/ct.zarr/", nml: "<things/>" } };
  assert.deepEqual(readSessionFile(`${JSON.stringify({ session, kept }, null, 2)}\n`), { token: token(session), kept });
  assert.throws(() => readSessionFile("#session=abc"), SyntaxError);
  assert.throws(() => readSessionFile(JSON.stringify(session)), /^Error: a session file holds session and kept alone/);
  assert.throws(() => readSessionFile(JSON.stringify({ session, kept: { Points: 1 } })), /kebab-case names/);
  const euler = { ...session, rotationMode: "euler" };
  assert.throws(() => readSessionFile(JSON.stringify({ session: euler, kept })), /Unknown rotation mode: euler/);
  const over = withPatch({ extensions: { unread: "x".repeat(SESSION_JSON_BYTE_LIMIT) } });
  assert.throws(
    () => readSessionFile(JSON.stringify({ session: over, kept: {} })),
    /^Error: its JSON is over the 32,768 bytes a session holds\.$/,
  );
});
