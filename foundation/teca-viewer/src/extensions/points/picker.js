function press(button, pressed) {
  button.classList.toggle("active", pressed);
  button.setAttribute("aria-pressed", String(pressed));
}

function fill(select, choices, placeholder) {
  const options = choices.map(([value, name]) => new Option(name, value === null ? "?" : String(value)));
  if (placeholder !== undefined) options.unshift(Object.assign(new Option(placeholder, ""), { hidden: true }));
  select.replaceChildren(...options);
}

export class TargetPicker {
  constructor(element, actions) {
    element.innerHTML = `
      <style>
        .points-pick { display: grid; grid-template-columns: 78px minmax(0, 1fr) auto; gap: var(--sp-2);
          align-items: center; }
        .points-pick > .segmented:last-child, .points-pick > .button-row { grid-column: 2 / 4; }
        .points-winding { display: flex; gap: 2px; align-items: center; }
        .points-winding button { padding: 5px 8px; }
        #points-winding { min-width: 3.5em; text-align: center; font-weight: 600; font-variant-numeric: tabular-nums; }
        #points-sides { grid-template-columns: 1fr 1fr auto; grid-auto-columns: auto; }
        #points-roles { grid-template-columns: minmax(0, 1fr) auto auto auto; }
        #points-roles > button { padding: 5px 3px; }
        #points-roles > div { display: grid; grid-template-columns: minmax(0, 1fr) auto; }
        #points-fiber { padding: 5px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
          text-align: left; }
        #points-fibers { width: 20px; padding: 5px 0; border-left: 0; background: var(--panel); color: var(--muted);
          border-color: var(--bevel-light) var(--bevel-dark) var(--bevel-dark) var(--bevel-light); }
        #points-fibers.active { background: var(--tint-accent); color: var(--accent);
          border-color: var(--bevel-dark) var(--bevel-light) var(--bevel-light) var(--bevel-dark); }
      </style>
      <div class="points-pick"><span class="control-name">Winding</span>
        <div class="points-winding">
          <button id="points-winding-down" type="button"
            title="The same role a winding down, or the air a gap down ([)">−</button>
          <span id="points-winding"></span>
          <button id="points-winding-up" type="button"
            title="The same role a winding up, or the air a gap up (])">+</button>
          <button id="points-winding-unknown" type="button" aria-pressed="false"
            title="The winding is unknown; pressed again, W0">?</button>
        </div>
        <button id="points-insert" type="button"
          title="Insert a winding here: add 1 to every winding from this one to its region's end, as one edit"
          >Insert</button>
      </div>
      <div class="points-pick"><span class="control-name">Region</span>
        <div class="button-row">
          <button id="points-new-region" type="button"
            title="W0 of a new region, above every region holding a class (Shift+N)">New region</button>
          <button id="points-join" type="button" aria-pressed="false"
            title="Press at a winding, then at the winding it is: its region moves there (Shift+J; Escape cancels)"
            >Join</button>
        </div>
      </div>
      <div class="points-pick"><span class="control-name">Side</span>
        <div id="points-sides" class="segmented">
          <button id="points-recto" type="button" title="The same role on the recto (T: the other side)">recto</button>
          <button id="points-verso" type="button" title="The same role on the verso (T: the other side)">verso</button>
          <button id="points-side-unknown" type="button" title="The side is unknown">?</button>
        </div>
        <select id="points-sheet"
          title="The sheet: S0 the side's own, S+1 or S-1 another sheet at a kollesis, S? unknown"></select>
      </div>
      <div class="points-pick"><span class="control-name">Role</span>
        <div id="points-roles" class="segmented">
          <div>
            <button id="points-fiber" type="button"
              title="This side's last fiber drawn, or a new one where it has none (Q)">Fiber</button>
            <select id="points-fibers" title="A fiber of this side, or a new one (N: a new one)"></select>
          </div>
          <button id="points-ink" type="button" title="This side's ink (A)">Ink</button>
          <button id="points-no-ink" type="button" title="This side's no ink (X)">No ink</button>
          <button id="points-air" type="button" title="The air between this winding and the next (Shift+A)">Air</button>
        </div>
      </div>`;
    const named = (...ids) => ids.map((id) => element.querySelector(`#points-${id}`));
    [this.down, this.winding, this.up, this.unknownWinding, this.insert, this.newRegion, this.join] = named(
      "winding-down",
      "winding",
      "winding-up",
      "winding-unknown",
      "insert",
      "new-region",
      "join",
    );
    [this.recto, this.verso, this.unknownSide, this.sheet, this.fiber, this.fibers, this.ink, this.noInk, this.air] =
      named("recto", "verso", "side-unknown", "sheet", "fiber", "fibers", "ink", "no-ink", "air");
    this.choosers = [
      this.unknownWinding,
      this.newRegion,
      this.recto,
      this.verso,
      this.unknownSide,
      this.sheet,
      this.fiber,
      this.fibers,
      this.ink,
      this.noInk,
      this.air,
    ];
    this.target = null;
    this.shown = null;
    const clicks = new Map([
      [this.down, () => actions.winding(-1)],
      [this.up, () => actions.winding(1)],
      [this.unknownWinding, () => actions.choose({ winding: this.target.winding === null ? 0 : null })],
      [this.insert, actions.insert],
      [this.newRegion, actions.region],
      [this.join, actions.join],
      [this.recto, () => actions.choose({ side: "recto" })],
      [this.verso, () => actions.choose({ side: "verso" })],
      [this.unknownSide, () => actions.choose({ side: null })],
      [this.fiber, () => actions.choose({ role: "fiber" })],
      [this.ink, () => actions.choose({ role: "ink" })],
      [this.noInk, () => actions.choose({ role: "no ink" })],
      [this.air, () => actions.choose({ air: true })],
    ]);
    element.addEventListener("click", (event) => clicks.get(event.target.closest("button"))?.());
    this.sheet.addEventListener("change", () =>
      actions.choose({ sheet: this.sheet.value === "?" ? null : Number(this.sheet.value) }),
    );
    this.fibers.addEventListener("change", () => actions.choose({ fiber: Number(this.fibers.value) }));
  }

  show({ ready, target, choices, color_RGB, insert, joining }) {
    this.target = target;
    const lists = JSON.stringify([choices.sheets, choices.fibers]);
    if (lists !== this.shown) {
      this.shown = lists;
      fill(this.sheet, choices.sheets);
      fill(this.fibers, choices.fibers, "");
    }
    const role = choices.tree !== null ? "tree" : target.tree !== null ? null : target.air ? "air" : target.role;
    this.winding.textContent = choices.winding;
    press(this.unknownWinding, target.winding === null);
    press(this.recto, role !== "air" && target.side === "recto");
    press(this.verso, role !== "air" && target.side === "verso");
    press(this.unknownSide, role !== "air" && target.side === null);
    this.sheet.value = role === "air" ? "" : target.sheet === null ? "?" : String(target.sheet);
    const pressed = role === "fiber" || role === "tree";
    if (role === "fiber") this.fiber.textContent = choices.fibers.find(([value]) => value === target.fiber)[1];
    else this.fiber.textContent = role === "tree" ? choices.tree : "Fiber";
    this.fibers.value = role === "fiber" ? String(target.fiber) : "";
    press(this.fiber, pressed);
    this.fibers.classList.toggle("active", pressed);
    press(this.ink, role === "ink");
    press(this.noInk, role === "no ink");
    press(this.air, role === "air");
    const underline = `inset 0 -3px 0 rgb(${color_RGB.map((value) => Math.round(255 * value))})`;
    for (const control of [this.fiber, this.fibers, this.ink, this.noInk, this.air])
      control.style.boxShadow = control.classList.contains("active") ? underline : "";
    for (const control of [...this.choosers, this.join]) control.disabled = !ready;
    press(this.join, joining);
    this.down.disabled = this.up.disabled = !ready || target.winding === null;
    this.insert.disabled = !insert;
  }
}
