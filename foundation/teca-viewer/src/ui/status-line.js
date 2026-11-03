import { STATUS_HOLD_MS } from "../config.js";
import { setStatusText } from "./dom.js";

export function createStatusLine(element) {
  let holdTimer = null;
  let held = null;
  let ambientMessage = null;

  function write(message) {
    element.title = message;
    setStatusText(
      element,
      message.replace(/https?:\/\/[^\s]+/g, (url) => url.split(/[?#]/)[0].split("/").filter(Boolean).at(-1)),
    );
  }

  function release() {
    holdTimer = null;
    held = null;
    if (ambientMessage !== null) write(ambientMessage);
  }

  function setStatus(message) {
    write(message);
    held = message;
    clearTimeout(holdTimer);
    holdTimer = setTimeout(release, STATUS_HOLD_MS);
  }

  function clearStatus(message) {
    if (held !== message) return;
    clearTimeout(holdTimer);
    release();
  }

  function setAmbientStatus(message) {
    ambientMessage = message;
    if (holdTimer === null) write(message);
  }

  return { setStatus, clearStatus, setAmbientStatus };
}
