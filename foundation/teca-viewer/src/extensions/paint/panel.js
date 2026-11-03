export const ICON = `<svg viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M16.5 3.5 9.5 11l-1-1 7-7.5z"/><path d="M8.5 10c-2 0-3.5 1.3-3.5 3.2 0 1.2-.8 2.3-2 2.8 3 1.2 7.2.6 7.5-3.6z"/></svg>`;
export const DEFAULT_RADIUS = 4;
export const MAX_RADIUS = 256;
const DEFAULT_GATE = 64;
export const AXES = ["z", "y", "x"];

export function paintPanel() {
  const panel = document.createElement("section");
  panel.id = "paint-section";
  panel.className = "card source-dependent";
  panel.setAttribute("aria-label", "Paint");
  panel.innerHTML = `
    <style>
      .paint-classes { grid-auto-flow: row; grid-template-columns: repeat(5, minmax(0, 1fr)); }
      .paint-radii { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 6px; }
      .paint-gate-value { width: 4.5em; }
    </style>
    <h2>Paint</h2>
    <div id="paint-classes" class="segmented paint-classes"></div>
    <label><span class="control-name">Radius</span>
      <span class="range-pair">
        <input id="paint-radius" type="range" min="1" max="${MAX_RADIUS}" step="1" value="${DEFAULT_RADIUS}" />
        <input id="paint-radius-number" type="number" min="1" max="${MAX_RADIUS}" step="1" value="${DEFAULT_RADIUS}" />
      </span>
    </label>
    <label class="checkbox-row"><input id="paint-axes" type="checkbox" /> Radius per axis, Z Y X</label>
    <div id="paint-radii" class="paint-radii" hidden>${AXES.map(
      (axis) => `<input id="paint-radius-${axis}" type="number" min="1" max="${MAX_RADIUS}" step="1"
        value="${DEFAULT_RADIUS}" aria-label="Radius along ${axis}" />`,
    ).join("")}</div>
    <label class="checkbox-row"><input id="paint-erase" type="checkbox" /> Erase</label>
    <label class="checkbox-row"><input id="paint-fill" type="checkbox" /> Fill a region with a click</label>
    <label class="checkbox-row"><input id="paint-gate" type="checkbox" /> Only where the CT is at least
      <input id="paint-gate-value" class="paint-gate-value" type="number" min="0" max="255" step="1"
        value="${DEFAULT_GATE}" aria-label="Gate on the CT's 0-255 display scale" /></label>
    <div class="button-row">
      <button id="paint-tool" type="button">Paint</button>
      <button id="paint-undo" type="button">Undo</button>
      <button id="paint-redo" type="button">Redo</button>
      <button id="paint-clear" type="button">Clear all</button>
    </div>
    <div class="button-row">
      <button id="paint-save" type="button">Save</button>
      <button id="paint-load" type="button">Load…</button>
    </div>
    <input id="paint-file" type="file" accept=".zip" hidden />
    <p id="paint-note" class="readout-line" role="status"></p>
    <div class="button-row"><button id="paint-here" type="button" hidden>Paint here</button></div>`;
  return panel;
}
