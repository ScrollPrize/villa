// The session travels in the address's #, which no request sends to its host.

import {
  DEFAULT_CAMERA_ORIENTATION,
  DEFAULT_ZOOM,
  DISPLAY_GAMMA_RANGE,
  HELD_SLICE_STEP,
  MAX_HELD_SLICE_STEP,
  MAX_ZOOM,
  MIN_HELD_SLICE_STEP,
  MIN_ZOOM,
  OVERLAY_LINE_WIDTH_RANGE,
  SESSION_JSON_BYTE_LIMIT,
  SESSION_PUBLISH_DELAY_MS,
  SURFACE_NORMAL_OFFSET_LIMIT,
  SURFACE_NORMAL_STEP_LIMITS,
  SURFACE_ZOOM_LIMITS,
} from "./config.js";
import { IDENTITY_ROTATION } from "./math.js";
import { ctName } from "./storage.js";
import { defaultSurfaceLayout, validateSurfaceLayout } from "./surface/layout.js";
import { defaultSurfaceVolumetricSettings, validateSurfaceVolumetricSettings } from "./surface/volumetric.js";

const QUATERNION_NORM_TOLERANCE = 1e-6;
const OPENED_KEPT = "teca-session-kept";
const SESSION_KEYS = [
  "sources",
  "activeSource",
  "plane",
  "activeViewport",
  "fullscreenViewport",
  "center_ZYX",
  "zoom",
  "manualScaleIndex",
  "manualCoarsestScaleIndex",
  "selectedChannel",
  "rotationMode",
  "rotation",
  "paneRotations",
  "cameraOrientation",
  "heldSliceStep",
  "displayBlack",
  "displayWhite",
  "displayGamma",
  "overlays",
  "overlayLineWidth",
  "surfaceView",
  "tabs",
  "extensions",
];
const NAME = /^[a-z]+(?:-[a-z]+)*$/;

export function createViewerSession() {
  return {
    sources: [],
    activeSource: null,
    plane: "xy",
    activeViewport: "xy",
    fullscreenViewport: null,
    center_ZYX: [0, 0, 0],
    zoom: DEFAULT_ZOOM,
    manualScaleIndex: null,
    manualCoarsestScaleIndex: null,
    selectedChannel: 0,
    rotationMode: "frame",
    rotation: [...IDENTITY_ROTATION],
    paneRotations: {
      xy: [...IDENTITY_ROTATION],
      zy: [...IDENTITY_ROTATION],
      xz: [...IDENTITY_ROTATION],
    },
    cameraOrientation: [...DEFAULT_CAMERA_ORIENTATION],
    heldSliceStep: HELD_SLICE_STEP,
    displayBlack: 0,
    displayWhite: 255,
    displayGamma: 1,
    overlays: [],
    overlayLineWidth: 2,
    surfaceView: {
      path: null,
      primarySource: null,
      center_UV: [0, 0],
      zoom: 1,
      normalOffset: 0,
      normalStep: 0.25,
      normalSign: 1,
      volumetric: defaultSurfaceVolumetricSettings(),
      layout: defaultSurfaceLayout(),
    },
    tabs: { left: "data", right: null },
    extensions: {},
  };
}

export function sessionSnapshot(session) {
  return Object.fromEntries(SESSION_KEYS.map((key) => [key, structuredClone(session[key])]));
}

function nonFinite(value) {
  return !Number.isFinite(value);
}

function isPlainObject(value) {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

function hasExactKeys(value, keys) {
  return isPlainObject(value) && Object.keys(value).sort().join("\0") === [...keys].sort().join("\0");
}

function isNonemptyString(value) {
  return typeof value === "string" && value.length > 0;
}

function validateUnitQuaternion(value, name) {
  if (!Array.isArray(value) || value.length !== 4 || value.some(nonFinite)) {
    throw new Error(`Session ${name} must contain four finite quaternion values.`);
  }
  const norm = Math.hypot(...value);
  if (Math.abs(norm - 1) > QUATERNION_NORM_TOLERANCE) {
    throw new Error(
      `Session ${name} must be a unit quaternion within ${QUATERNION_NORM_TOLERANCE}; received norm ${norm}.`,
    );
  }
}

function validateSurfaceView({ surfaceView, sources, overlays, activeViewport, fullscreenViewport }) {
  const selectedOverlay =
    surfaceView?.path === null
      ? null
      : overlays.find(
          (overlay) => overlay.path === surfaceView?.path && overlay.primarySource === surfaceView?.primarySource,
        );
  if (
    !hasExactKeys(surfaceView, [
      "path",
      "primarySource",
      "center_UV",
      "zoom",
      "normalOffset",
      "normalStep",
      "normalSign",
      "volumetric",
      "layout",
    ]) ||
    (surfaceView.path !== null && !isNonemptyString(surfaceView.path)) ||
    (surfaceView.path === null) !== (surfaceView.primarySource === null) ||
    (surfaceView.primarySource !== null && !sources.includes(surfaceView.primarySource)) ||
    (surfaceView.path !== null && selectedOverlay === undefined) ||
    (fullscreenViewport === "surface" && selectedOverlay?.visible !== true) ||
    (activeViewport === "surface" && fullscreenViewport !== "surface") ||
    (fullscreenViewport === "surface" &&
      activeViewport !== "surface" &&
      (surfaceView.layout?.mode === "solo" || !surfaceView.layout?.planes?.includes(activeViewport))) ||
    !Array.isArray(surfaceView.center_UV) ||
    surfaceView.center_UV.length !== 2 ||
    surfaceView.center_UV.some(nonFinite) ||
    !Number.isFinite(surfaceView.zoom) ||
    surfaceView.zoom < SURFACE_ZOOM_LIMITS[0] ||
    surfaceView.zoom > SURFACE_ZOOM_LIMITS[1] ||
    !Number.isFinite(surfaceView.normalOffset) ||
    surfaceView.normalOffset < -SURFACE_NORMAL_OFFSET_LIMIT ||
    surfaceView.normalOffset > SURFACE_NORMAL_OFFSET_LIMIT ||
    !Number.isFinite(surfaceView.normalStep) ||
    surfaceView.normalStep < SURFACE_NORMAL_STEP_LIMITS[0] ||
    surfaceView.normalStep > SURFACE_NORMAL_STEP_LIMITS[1] ||
    !new Set([-1, 1]).has(surfaceView.normalSign)
  ) {
    throw new Error("Session surface view is malformed.");
  }
  validateSurfaceVolumetricSettings(surfaceView.volumetric);
  validateSurfaceLayout(surfaceView.layout);
}

function validateSession(session) {
  if (!isPlainObject(session)) throw new Error("Session must be a JSON object.");
  if (!hasExactKeys(session, SESSION_KEYS)) {
    throw new Error(`Session keys must be exactly ${SESSION_KEYS.join(", ")}.`);
  }
  if (!Array.isArray(session.sources) || !Array.isArray(session.overlays)) {
    throw new Error("Session sources and overlays must be arrays.");
  }
  for (const source of session.sources) {
    if (!isNonemptyString(source) || !/^https?:\/\//.test(source)) {
      throw new Error(`Every session source is an HTTP(S) URL: ${JSON.stringify(source)}.`);
    }
  }
  if (new Set(session.sources).size !== session.sources.length) {
    throw new Error("Session sources must be unique.");
  }
  if (session.activeSource !== null && !session.sources.includes(session.activeSource)) {
    throw new Error(`Active source ${session.activeSource} is absent from session sources.`);
  }
  if (!Array.isArray(session.center_ZYX) || session.center_ZYX.length !== 3 || session.center_ZYX.some(nonFinite)) {
    throw new Error("Session center_ZYX must contain three finite coordinates.");
  }
  if (!new Set(["frame", "pane"]).has(session.rotationMode)) {
    throw new Error(`Unknown rotation mode: ${session.rotationMode}.`);
  }
  validateUnitQuaternion(session.rotation, "rotation");
  if (!hasExactKeys(session.paneRotations, ["xy", "zy", "xz"])) {
    throw new Error("Session paneRotations must contain exactly xy, zy, and xz.");
  }
  for (const plane of ["xy", "zy", "xz"]) {
    validateUnitQuaternion(session.paneRotations[plane], `paneRotations.${plane}`);
  }
  validateUnitQuaternion(session.cameraOrientation, "cameraOrientation");
  for (const key of ["zoom", "displayBlack", "displayWhite", "displayGamma", "overlayLineWidth"]) {
    if (!Number.isFinite(session[key])) throw new Error(`Session ${key} must be finite.`);
  }
  if (session.zoom < MIN_ZOOM || session.zoom > MAX_ZOOM) {
    throw new Error(`Session zoom must be inside ${MIN_ZOOM}..${MAX_ZOOM}.`);
  }
  if (
    session.displayBlack < 0 ||
    session.displayBlack > 254 ||
    session.displayWhite < 1 ||
    session.displayWhite > 255 ||
    session.displayWhite <= session.displayBlack
  ) {
    throw new Error("Session display bounds must satisfy 0 <= black < white <= 255.");
  }
  if (session.displayGamma < DISPLAY_GAMMA_RANGE[0] || session.displayGamma > DISPLAY_GAMMA_RANGE[1]) {
    throw new Error(`Session displayGamma must remain inside ${DISPLAY_GAMMA_RANGE.join("..")}.`);
  }
  if (
    session.overlayLineWidth < OVERLAY_LINE_WIDTH_RANGE[0] ||
    session.overlayLineWidth > OVERLAY_LINE_WIDTH_RANGE[1]
  ) {
    throw new Error(
      `Session overlayLineWidth ${session.overlayLineWidth} is outside ${OVERLAY_LINE_WIDTH_RANGE.join("..")}.`,
    );
  }
  if (!Number.isInteger(session.selectedChannel) || session.selectedChannel < 0) {
    throw new Error("Session selectedChannel must be a non-negative integer.");
  }
  for (const key of ["manualScaleIndex", "manualCoarsestScaleIndex"]) {
    if (session[key] !== null && (!Number.isInteger(session[key]) || session[key] < 0)) {
      throw new Error(`Session ${key} must be null or a non-negative integer.`);
    }
  }
  if (
    session.manualScaleIndex !== null &&
    session.manualCoarsestScaleIndex !== null &&
    session.manualScaleIndex > session.manualCoarsestScaleIndex
  ) {
    throw new Error("Session finest level must not be coarser than its coarsest level.");
  }
  if (
    !Number.isInteger(session.heldSliceStep) ||
    session.heldSliceStep < MIN_HELD_SLICE_STEP ||
    session.heldSliceStep > MAX_HELD_SLICE_STEP
  ) {
    throw new Error(`Session heldSliceStep must be an integer inside ${MIN_HELD_SLICE_STEP}..${MAX_HELD_SLICE_STEP}.`);
  }
  if (!new Set(["xy", "zy", "xz"]).has(session.plane)) throw new Error(`Unknown session plane: ${session.plane}.`);
  if (!new Set(["xy", "zy", "xz", "3d", "surface"]).has(session.activeViewport)) {
    throw new Error(`Unknown active viewport: ${session.activeViewport}.`);
  }
  if (
    session.fullscreenViewport !== null &&
    !new Set(["xy", "zy", "xz", "3d", "surface"]).has(session.fullscreenViewport)
  ) {
    throw new Error(`Unknown fullscreen viewport: ${session.fullscreenViewport}.`);
  }
  for (const overlay of session.overlays) {
    if (
      !hasExactKeys(overlay, ["path", "primarySource", "visible", "colorIndex"]) ||
      !isNonemptyString(overlay.path) ||
      !session.sources.includes(overlay.primarySource) ||
      typeof overlay.visible !== "boolean" ||
      !Number.isInteger(overlay.colorIndex) ||
      overlay.colorIndex < 0
    ) {
      throw new Error(`Session overlay is malformed: ${JSON.stringify(overlay)}.`);
    }
  }
  if (new Set(session.overlays.map((overlay) => overlay.path)).size !== session.overlays.length) {
    throw new Error("Session overlay paths must be unique.");
  }
  validateSurfaceView(session);
  const { tabs } = session;
  if (
    !hasExactKeys(tabs, ["left", "right"]) ||
    !["data", "view"].includes(tabs.left) ||
    (tabs.right !== null && !(typeof tabs.right === "string" && NAME.test(tabs.right)))
  )
    throw new Error(
      `Session tabs must be the left's data or view and the right's name or null: ${JSON.stringify(tabs)}.`,
    );
  if (!isPlainObject(session.extensions) || Object.keys(session.extensions).some((name) => !NAME.test(name)))
    throw new Error(
      `Session extensions must be an object keyed by kebab-case names: ${JSON.stringify(session.extensions)}.`,
    );
}

export function sessionToken(session) {
  const snapshot = sessionSnapshot(session);
  validateSession(snapshot);
  const encoder = new TextEncoder();
  const bytes = encoder.encode(JSON.stringify(snapshot));
  if (bytes.byteLength > SESSION_JSON_BYTE_LIMIT) {
    const { extensions, ...fields } = snapshot;
    const [name, size] = [...Object.entries(fields), ...Object.entries(extensions)]
      .map(([part, value]) => [part, encoder.encode(JSON.stringify(value)).byteLength])
      .reduce((largest, part) => (part[1] > largest[1] ? part : largest));
    const count = (value) => value.toLocaleString("en-US");
    const refused =
      `The link holds the view as it last fitted: this one is ${count(bytes.byteLength)} bytes, over ` +
      `${count(SESSION_JSON_BYTE_LIMIT)}, and ${name} take ${count(size)} of them.`;
    return { token: null, refused };
  }
  let binary = "";
  for (const byte of bytes) binary += String.fromCharCode(byte);
  return { token: btoa(binary).replaceAll("+", "-").replaceAll("/", "_").replaceAll("=", ""), refused: null };
}

export function deserializeSession(token) {
  if (typeof token !== "string" || token.length === 0) throw new Error("Session link is empty.");
  if (token.length > Math.ceil((SESSION_JSON_BYTE_LIMIT * 4) / 3) + 4) {
    throw new Error(`Session link exceeds the ${SESSION_JSON_BYTE_LIMIT}-byte decoded limit.`);
  }
  const base64 = token.replaceAll("-", "+").replaceAll("_", "/");
  const binary = atob(base64 + "=".repeat((4 - (base64.length % 4)) % 4));
  const bytes = Uint8Array.from(binary, (character) => character.charCodeAt(0));
  const session = JSON.parse(new TextDecoder().decode(bytes));
  validateSession(session);
  return session;
}

export function readSessionFile(text) {
  const file = JSON.parse(text);
  if (!isPlainObject(file) || !hasExactKeys(file, ["session", "kept"]))
    throw new Error(`a session file holds session and kept alone, not ${JSON.stringify(text.slice(0, 80))}.`);
  if (!isPlainObject(file.kept) || Object.keys(file.kept).some((name) => !NAME.test(name)))
    throw new Error(`its kept must be an object keyed by kebab-case names: ${JSON.stringify(file.kept)}.`);
  validateSession(file.session);
  const { token, refused } = sessionToken(file.session);
  if (refused !== null)
    throw new Error(`its JSON is over the ${SESSION_JSON_BYTE_LIMIT.toLocaleString("en-US")} bytes a session holds.`);
  return { token, kept: file.kept };
}

export function openedKept(stash) {
  const kept = stash.getItem(OPENED_KEPT);
  stash.removeItem(OPENED_KEPT);
  return kept === null ? {} : JSON.parse(kept);
}

export function linkedSession(address) {
  const token = new URLSearchParams(address.hash.slice(1)).get("session");
  if (token === null) return { session: createViewerSession(), refused: null };
  try {
    return { session: deserializeSession(token), refused: null };
  } catch (error) {
    return { session: createViewerSession(), refused: `Session link rejected: ${error.message}` };
  }
}

export function createSessionAddress({ state, setStatus, clearStatus, stash }) {
  let held = false;
  let burst = null;
  let last = null;
  let said = null;

  // A link that differs from the address in the fragment alone navigates within the same document, so the page reloads
  // to read its session.
  window.addEventListener("hashchange", () => location.reload());

  function publishSession() {
    const { token, refused } = sessionToken(state);
    if (refused !== null) {
      if (said === null) {
        said = refused;
        setStatus(said);
      }
      return refused;
    }
    if (said !== null) clearStatus(said);
    said = null;
    if (held) return null;
    const url = new URL((last ?? location).href);
    url.hash = `session=${token}`;
    url.searchParams.delete("volume");
    for (const link of url.searchParams.getAll("overlay"))
      if (state.overlays.some(({ path }) => link.startsWith(path) && /^(?:[/?#]|$)/.test(link.slice(path.length))))
        url.searchParams.delete("overlay", link);
    if (burst === null) history.replaceState(null, "", url);
    else last = url;
    clearTimeout(burst);
    burst = setTimeout(() => {
      burst = null;
      if (last !== null) history.replaceState(null, "", last);
      last = null;
    }, SESSION_PUBLISH_DELAY_MS);
    return null;
  }

  function sessionAddress() {
    return (last ?? location).href;
  }

  function sessionFile(kept) {
    const { session } = linkedSession(new URL(sessionAddress()));
    const name = session.activeSource === null ? "viewer" : ctName(session.activeSource);
    return { name: `${name}-session.json`, text: `${JSON.stringify({ session, kept }, null, 2)}\n` };
  }

  function openSessionFile(text) {
    const { token, kept } = readSessionFile(text);
    stash.setItem(OPENED_KEPT, JSON.stringify(kept));
    const url = new URL(location.href);
    url.hash = `session=${token}`;
    for (const key of ["volume", "overlay"]) url.searchParams.delete(key);
    history.pushState(null, "", url);
    location.reload();
  }

  function holdPublication(hold) {
    held = hold;
  }

  function publicationHeld() {
    return held;
  }

  return { publishSession, sessionAddress, sessionFile, openSessionFile, holdPublication, publicationHeld };
}
