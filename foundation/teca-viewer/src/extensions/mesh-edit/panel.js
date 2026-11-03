const MIN_RADIUS = 8;
const MAX_RADIUS = 1024;
const RESIZE = 1.25;

export class MeshEditPanel {
  constructor(actions, modes, tool) {
    this.actions = actions;
    this.modes = modes;
    this.tool = tool;
    const panel = document.createElement("section");
    panel.id = "mesh-edit-section";
    panel.className = "card source-dependent";
    panel.setAttribute("aria-label", "Mesh edits");
    panel.innerHTML = `<h2>Mesh edits</h2>
      <form id="mesh-edit-form" class="overlay-form">
        <select id="mesh-edit-surface" aria-label="Surface to edit"></select>
        <button id="mesh-edit-open" type="submit">Edit</button>
      </form>
      <label><span class="control-name">Radius</span>
        <span class="range-pair">
          <input id="mesh-edit-radius" type="range" min="${MIN_RADIUS}" max="${MAX_RADIUS}" step="1"
            value="${tool.radius}" />
          <input id="mesh-edit-radius-number" type="number" min="${MIN_RADIUS}" max="${MAX_RADIUS}" step="1"
            value="${tool.radius}" />
        </span>
      </label>
      <div class="button-row">
        ${modes.map(({ kind, name }) => `<button id="mesh-edit-${kind}" type="button">${name}</button>`).join("")}
      </div>
      <div class="button-row">
        <button id="mesh-edit-undo" type="button">Undo</button>
        <button id="mesh-edit-redo" type="button">Redo</button>
      </div>
      <div class="button-row">
        <button id="mesh-edit-save" type="button">Save</button>
        <button id="mesh-edit-discard" type="button">Discard</button>
      </div>
      <p id="mesh-edit-note" class="readout-line" role="status"></p>`;
    this.element = panel;
    this.surfaceSelect = panel.querySelector("#mesh-edit-surface");
    [this.radiusInput, this.radiusNumber] = panel.querySelectorAll("input");
    const [openButton, ...buttons] = panel.querySelectorAll("button");
    this.openButton = openButton;
    this.modeButtons = buttons.slice(0, modes.length);
    [this.undoButton, this.redoButton, this.saveButton, this.discardButton] = buttons.slice(modes.length);
    this.note = panel.querySelector("#mesh-edit-note");
    panel.querySelector("#mesh-edit-form").addEventListener("submit", (event) => {
      event.preventDefault();
      actions.edit(this.surfaceSelect.value);
    });
    for (const [position, button] of this.modeButtons.entries())
      button.addEventListener("click", () => actions.take(modes[position]));
    this.undoButton.addEventListener("click", () => actions.undo());
    this.redoButton.addEventListener("click", () => actions.redo());
    this.saveButton.addEventListener("click", () => actions.save());
    this.discardButton.addEventListener("click", () => actions.discard(this.surfaceSelect.value));
    this.radiusInput.addEventListener("input", () => actions.resize(Number(this.radiusInput.value)));
    this.radiusNumber.addEventListener("change", () => actions.resize(Number(this.radiusNumber.value)));
  }

  radius(radius) {
    this.radiusInput.value = String(radius);
    return Number(this.radiusInput.value);
  }

  list(urls, edited) {
    const select = this.surfaceSelect;
    const chosen = [select.value, edited].find((url) => urls.includes(url)) ?? urls[0] ?? "";
    select.replaceChildren(
      ...urls.map((url) => {
        const option = new Option(url.split("/").at(-1), url);
        option.title = url;
        return option;
      }),
    );
    if (urls.length === 0) select.append(new Option("Add a surface in the Surfaces list", ""));
    select.value = chosen;
  }

  choose(url) {
    this.surfaceSelect.value = url;
  }

  show({ busy, ct, edited, editable, active, mode, undo, redo, note }) {
    const none = this.surfaceSelect.value === "";
    this.surfaceSelect.disabled = busy || none;
    this.openButton.disabled = busy || !ct || none;
    for (const [position, button] of this.modeButtons.entries()) {
      button.classList.toggle("active", active && this.modes[position] === mode);
      button.disabled = busy || !editable;
    }
    this.undoButton.disabled = busy || !undo;
    this.redoButton.disabled = busy || !redo;
    this.saveButton.disabled = busy || !edited;
    this.discardButton.disabled = busy || !ct;
    if (document.activeElement !== this.radiusNumber) this.radiusNumber.value = this.radiusInput.value;
    this.note.textContent = note;
  }

  keys() {
    const { actions, tool } = this;
    const plain = { shift: false, command: false, alt: false };
    const moving = { ...plain, tool };
    const stepping = { ...moving, repeats: true };
    return [
      {
        description: "Mesh edits: take the tool or put it away",
        code: "Digit8",
        ...plain,
        run: actions.tool,
      },
      ...this.modes.map((next) => ({
        description: `Mesh edits: ${next.name.toLowerCase()}`,
        code: next.code,
        ...moving,
        run: () => actions.take(next),
      })),
      {
        description: "Mesh edits: smaller radius",
        code: "BracketLeft",
        ...stepping,
        run: () => actions.resize(Math.min(tool.radius - 1, Math.round(tool.radius / RESIZE))),
      },
      {
        description: "Mesh edits: larger radius",
        code: "BracketRight",
        ...stepping,
        run: () => actions.resize(Math.max(tool.radius + 1, Math.round(tool.radius * RESIZE))),
      },
      {
        description: "Mesh edits: take the last edit back",
        code: "KeyZ",
        ...stepping,
        command: true,
        run: () => this.undoButton.click(),
      },
      {
        description: "Mesh edits: do the edit taken back again",
        code: "KeyZ",
        ...stepping,
        shift: true,
        command: true,
        run: () => this.redoButton.click(),
      },
    ];
  }
}
