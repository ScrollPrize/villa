import { TargetPicker } from "./picker.js";
import { RunList } from "./run-list.js";
import { counted } from "./view.js";

export const ICON = `<svg viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><circle cx="6" cy="5" r="2"/><circle cx="6" cy="15" r="2"/><circle cx="14" cy="10" r="2"/><path d="M6 7v6M8 5.7 12.3 9M8 14.3 12.3 11"/></svg>`;

function show(input, value) {
  input.defaultValue = value;
  if (document.activeElement !== input) input.value = value;
}

export class PointsPanel {
  constructor(actions, pen) {
    this.actions = actions;
    this.pen = pen;
    const panel = document.createElement("section");
    panel.id = "points-section";
    panel.className = "card source-dependent";
    panel.setAttribute("aria-label", "Points");
    panel.innerHTML = `
      <style>
        #points-picker { display: grid; gap: var(--sp-2); }
        .points-runs { display: grid; gap: 2px; }
        .points-head, .points-scribble { display: flex; justify-content: space-between; }
        .points-head span + span, .points-scribble span + span { font-weight: normal; letter-spacing: normal;
          text-transform: none; }
        .points-scribble { gap: var(--sp-3); }
        #points-scribble-place { margin-left: auto; }
        #points-mode { margin: -4px 0; }
        #points-mode button { padding: 1px 8px; line-height: 12px; }
        #points-mark { display: grid; grid-template-columns: 78px minmax(0, 1fr); gap: var(--sp-2);
          align-items: center; }
      </style>
      <h2 class="points-head"><span>Points</span><span id="points-mode" class="segmented"
        ><button id="points-draw" type="button" aria-pressed="true"
          title="Draw: a press places or moves nodes, and the picker sets the next scribble's class (Escape)"
          >Draw</button><button id="points-edit" type="button" aria-pressed="false"
          title="Edit: a press picks a scribble, Enter or Ctrl+press keeps more, and the picker moves them (E; Shift+E: its class)"
          >Edit</button></span></h2>
      <div class="button-row">
        <button id="points-tool" type="button">Points</button>
        <button id="points-undo" type="button">Undo</button>
        <button id="points-redo" type="button">Redo</button>
      </div>
      <div id="points-picker"></div>
      <div id="points-mark"><span class="control-name">Mark</span>
        <button id="points-mark-kollesis" type="button" aria-pressed="false"
          title="Tag each node placed a kollesis while pressed (Shift+K; Escape releases)">Kollesis</button>
      </div>
      <label class="checkbox-row"><input id="points-only" type="checkbox" /> Only this winding</label>
      <label class="checkbox-row" title="Centre the panes on each node placed or selected (Shift+O)"
        ><input id="points-follow" type="checkbox" /> Follow each node placed or selected</label>
      <label class="checkbox-row" title="Draw the classes in the 3D pane, each class's scribbles joined by dashed links"
        ><input id="points-links" type="checkbox" /> Links in 3D</label>
      <label><span class="control-name">Joined</span>
        <select id="points-joined">
          <option value="separate">Separate</option>
          <option value="connected">Connected</option>
        </select>
      </label>
      <label><span class="control-name">Spacing</span>
        <input id="points-spacing" type="number" min="0.25" max="256" step="0.25" />
      </label>
      <label><span class="control-name">Depth</span>
        <input id="points-depth" type="number" min="0" max="256" step="0.5" />
      </label>
      <h3 class="points-scribble"><span>Selected scribble</span><span id="points-scribble-place"
        title="Its place among its class's scribbles in the order drawn (Shift+; and Shift+' step)"></span
        ><span id="points-scribble-count"></span></h3>
      <div class="button-row">
        <button id="points-delete" type="button"
          title="Delete the selected scribble, or with none an empty class (Delete; Shift+Delete: the node)"
          >Delete</button>
      </div>
      <label><span class="control-name">Note</span><input id="points-comment" type="text" maxlength="256" /></label>
      <label title="The class the picker shows, which S goes to: the selected node in it, else its scribble drawn last"
        ><span class="control-name">Class</span><select id="points-class"></select></label>
      <label><span class="control-name">Name</span><input id="points-name" type="text" maxlength="128" /></label>
      <div class="button-row">
        <button id="points-new-class" type="button"
          title="A class outside the roles, named in Name: the next scribble's, or in Edit the picked one's (C)"
          >New class</button>
        <button id="points-delete-class" type="button" title="Take the class out with all its scribbles"
          >Delete class</button>
      </div>
      <div class="button-row">
        <button id="points-save" type="button">Save</button>
        <button id="points-open" type="button">Open…</button>
        <button id="points-clear" type="button">Clear all</button>
      </div>
      <input id="points-file" type="file" accept=".nml,.zip" hidden />
      <p id="points-note" class="readout-line" role="status"></p>
      <div id="points-marked" class="points-runs" hidden></div>`;
    this.element = panel;
    const named = (...ids) => ids.map((id) => panel.querySelector(`#points-${id}`));
    [this.onlyInput, this.followInput, this.linksInput, this.joinedSelect] = named("only", "follow", "links", "joined");
    [this.spacingInput, this.depthInput, this.commentInput, this.classSelect] = named(
      "spacing",
      "depth",
      "comment",
      "class",
    );
    [this.nameInput, this.fileInput, this.note] = named("name", "file", "note");
    [this.drawButton, this.editButton, this.toolButton, this.undoButton, this.redoButton] = named(
      "draw",
      "edit",
      "tool",
      "undo",
      "redo",
    );
    this.picker = new TargetPicker(...named("picker"), actions);
    this.runList = new RunList(...named("marked"), actions.run);
    [this.markKollesisButton, this.scribblePlace, this.scribbleCount, this.deleteButton] = named(
      "mark-kollesis",
      "scribble-place",
      "scribble-count",
      "delete",
    );
    [this.newClassButton, this.deleteClassButton, this.saveButton, this.openButton, this.clearButton] = named(
      "new-class",
      "delete-class",
      "save",
      "open",
      "clear",
    );
    this.shownClasses = null;
    this.wire();
  }

  joined() {
    return this.joinedSelect.value === "connected";
  }

  wire() {
    const actions = this.actions;
    this.drawButton.addEventListener("click", () => actions.mode(false));
    this.editButton.addEventListener("click", () => actions.mode(true));
    this.toolButton.addEventListener("click", () => actions.tool());
    this.undoButton.addEventListener("click", () => actions.undo());
    this.redoButton.addEventListener("click", () => actions.redo());
    this.markKollesisButton.addEventListener("click", () => actions.mark("kollesis"));
    this.deleteButton.addEventListener("click", () => actions.remove());
    this.newClassButton.addEventListener("click", () => actions.newClass());
    this.deleteClassButton.addEventListener("click", () => actions.deleteClass());
    this.saveButton.addEventListener("click", () => actions.save());
    this.openButton.addEventListener("click", () => this.fileInput.click());
    this.clearButton.addEventListener("click", () => actions.clear());
    // Escape puts the old value back first, so the blur fires no change.
    const typed = (input, run) => {
      input.addEventListener("change", () => {
        if (run(input.value) === false) input.value = input.defaultValue;
      });
      input.addEventListener("keydown", (event) => {
        if (event.key === "Escape") input.value = input.defaultValue;
        if (event.key === "Enter" || event.key === "Escape") input.blur();
      });
    };
    typed(this.commentInput, actions.comment);
    typed(this.nameInput, actions.name);
    this.classSelect.addEventListener("change", () => actions.chooseClass(Number(this.classSelect.value)));
    this.onlyInput.addEventListener("change", () => actions.only(this.onlyInput.checked));
    this.followInput.addEventListener("change", () => actions.follow(this.followInput.checked));
    this.linksInput.addEventListener("change", () => actions.links(this.linksInput.checked));
    this.spacingInput.addEventListener("change", () => {
      this.spacingInput.value = String(actions.spacing(this.spacingInput.valueAsNumber));
    });
    this.depthInput.addEventListener("change", () => {
      this.depthInput.value = String(actions.depth(this.depthInput.valueAsNumber));
    });
    this.fileInput.addEventListener("change", () => {
      const [chosen] = this.fileInput.files;
      this.fileInput.value = "";
      if (chosen !== undefined) actions.open(chosen);
    });
  }

  show(view) {
    const { ready, tree, tags, scribble } = view;
    this.toolButton.classList.toggle("active", view.active);
    for (const [button, pressed] of [
      [this.drawButton, !view.editing],
      [this.editButton, view.editing],
    ]) {
      button.classList.toggle("active", pressed);
      button.setAttribute("aria-pressed", String(pressed));
      button.disabled = !view.active;
    }
    for (const button of [this.toolButton, this.newClassButton, this.openButton]) button.disabled = !ready;
    this.undoButton.disabled = !view.undo;
    this.redoButton.disabled = !view.redo;
    this.saveButton.disabled = view.classes.length === 0;
    this.clearButton.disabled = !view.clear;
    this.onlyInput.checked = view.only;
    this.followInput.checked = view.follow;
    this.linksInput.checked = view.links;
    show(this.spacingInput, String(view.spacing));
    show(this.depthInput, String(view.depth));
    this.deleteClassButton.disabled = tree === null;
    this.nameInput.disabled = tree === null;
    show(this.nameInput, tree?.name ?? "");
    this.markKollesisButton.setAttribute("aria-pressed", String(view.mark === "kollesis"));
    this.markKollesisButton.disabled = !ready;
    this.scribblePlace.textContent = scribble === null ? "" : `${scribble.place} of ${scribble.count}`;
    const nodes = scribble === null ? "" : counted(scribble.nodes, "node");
    this.scribbleCount.textContent = view.empty ? "empty class" : nodes;
    this.deleteButton.disabled = scribble === null && !view.empty;
    this.commentInput.disabled = tags === null;
    show(this.commentInput, tags?.text ?? "");
    const classes = JSON.stringify(view.classes);
    if (classes !== this.shownClasses) {
      this.shownClasses = classes;
      const groups = view.classes.map(({ name, trees }) => {
        const group = Object.assign(document.createElement("optgroup"), { label: name });
        group.append(...trees.map(({ id, label }) => new Option(label, String(id))));
        return group;
      });
      this.classSelect.replaceChildren(...groups);
    }
    this.classSelect.value = tree === null ? "" : String(tree.id);
    this.classSelect.disabled = view.classes.length === 0;
    this.picker.show(view);
    this.note.textContent = view.note;
    this.runList.show(view.runs, view.runRow, view.currentRun);
  }

  keys() {
    const actions = this.actions;
    const plain = { shift: false, command: false, alt: false };
    const placing = { ...plain, tool: this.pen };
    const shifted = { ...placing, shift: true };
    const stepping = { ...placing, repeats: true };
    return [
      {
        description: "Points: take the tool or put it away",
        code: "Digit6",
        ...plain,
        run: () => this.toolButton.click(),
      },
      { description: "Points: back to the class before", code: "Tab", ...placing, run: actions.previous },
      { description: "Points: this side's ink", code: "KeyA", ...placing, run: () => actions.role("ink") },
      { description: "Points: this side's no ink", code: "KeyX", ...placing, run: () => actions.role("no ink") },
      {
        description: "Points: this side's last fiber drawn",
        code: "KeyQ",
        ...placing,
        run: () => actions.role("fiber"),
      },
      { description: "Points: a new fiber on this side", code: "KeyN", ...placing, run: actions.newFiber },
      { description: "Points: the same role on the other side", code: "KeyT", ...placing, run: actions.otherSide },
      {
        description: "Points: the air between this winding and the next, or from air the class it was chosen from",
        code: "KeyA",
        ...shifted,
        run: actions.air,
      },
      {
        description: "Points: the same role a winding down, or from air the gap below",
        code: "BracketLeft",
        ...stepping,
        run: () => actions.winding(-1),
      },
      {
        description: "Points: the same role a winding up, or from air the gap above",
        code: "BracketRight",
        ...stepping,
        run: () => actions.winding(1),
      },
      {
        description: "Points: only this winding, or all",
        code: "KeyW",
        ...placing,
        run: () => this.onlyInput.click(),
      },
      {
        description: "Points: follow each node placed or selected, or not",
        code: "KeyO",
        ...shifted,
        run: () => this.followInput.click(),
      },
      {
        description: "Points: press or release Mark's Kollesis, tagging each node placed",
        code: "KeyK",
        ...shifted,
        run: () => actions.mark("kollesis"),
      },
      {
        description: "Points: Edit, where a press picks a scribble and the keys change its class, or Draw",
        code: "KeyE",
        ...placing,
        run: actions.switchMode,
      },
      {
        description: "Points: Edit with the selected node's whole class picked, or switch the pick to its scribble",
        code: "KeyE",
        ...shifted,
        run: actions.switchPick,
      },
      {
        description: "Points: let go of Join pressed once and of Mark, and go back to Draw",
        code: "Escape",
        ...placing,
        run: actions.release,
      },
      {
        description: "Points: W0 of a new region, above every region holding a class",
        code: "KeyN",
        ...shifted,
        run: actions.region,
      },
      {
        description: "Points: Join, pressed at a winding of the region to move and again at the winding it is",
        code: "KeyJ",
        ...shifted,
        run: actions.join,
      },
      {
        description: "Points: write a note on the selected node",
        code: "Slash",
        ...shifted,
        run: () => this.commentInput.focus(),
      },
      {
        description: "Points: new class, the next scribble's, or in Edit the picked scribble's",
        code: "KeyC",
        ...placing,
        run: () => this.newClassButton.click(),
      },
      {
        description: "Points: flag the selected node as a branchpoint, or unflag it",
        code: "KeyB",
        ...placing,
        run: actions.branchpoint,
      },
      {
        description: "Points: back to the latest branchpoint, unflagging it",
        code: "KeyJ",
        ...placing,
        run: actions.jump,
      },
      {
        description: "Points: centre on the selected node, the Surface too, or go to the class the picker shows",
        code: "KeyS",
        ...placing,
        run: () => actions.stepScribble(0),
      },
      {
        description: "Points: previous node of the class",
        code: "Semicolon",
        ...stepping,
        run: () => actions.step(-1),
      },
      { description: "Points: next node of the class", code: "Quote", ...stepping, run: () => actions.step(1) },
      {
        description: "Points: previous scribble of the class the picker shows, in the order drawn, or its last",
        code: "Semicolon",
        ...stepping,
        shift: true,
        run: () => actions.stepScribble(-1),
      },
      {
        description: "Points: next scribble of the class the picker shows, in the order drawn, or its first",
        code: "Quote",
        ...stepping,
        shift: true,
        run: () => actions.stepScribble(1),
      },
      {
        description: "Points: in Edit, keep the picked scribble, which a change of class then moves with the rest kept",
        code: "Enter",
        ...placing,
        run: actions.keep,
      },
      {
        description: "Points: previous class with nodes in the Class list, at its last scribble",
        code: "Comma",
        ...stepping,
        shift: true,
        run: () => actions.stepClass(-1),
      },
      {
        description: "Points: next class with nodes in the Class list, at its last scribble",
        code: "Period",
        ...stepping,
        shift: true,
        run: () => actions.stepClass(1),
      },
      { description: "Points: previous marked run", code: "Comma", ...stepping, run: () => actions.stepRun(-1) },
      { description: "Points: next marked run", code: "Period", ...stepping, run: () => actions.stepRun(1) },
      {
        description: "Points: delete the selected scribble, or with none an empty class",
        code: "Delete",
        ...placing,
        run: actions.remove,
      },
      {
        description: "Points: delete the selected scribble, or with none an empty class",
        code: "Backspace",
        ...placing,
        run: actions.remove,
      },
      { description: "Points: delete the selected node", code: "Delete", ...shifted, run: actions.removeNode },
      { description: "Points: delete the selected node", code: "Backspace", ...shifted, run: actions.removeNode },
      { description: "Points: undo", code: "KeyZ", ...stepping, command: true, run: () => this.undoButton.click() },
      {
        description: "Points: redo",
        code: "KeyZ",
        ...stepping,
        shift: true,
        command: true,
        run: () => this.redoButton.click(),
      },
    ];
  }
}
