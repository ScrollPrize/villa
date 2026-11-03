import { counted } from "./view.js";

const RUN_ROW_PX = 26;
const RUNS_SHOWN = 12;
const RUNS_BEYOND = 6;

export class RunList {
  constructor(element, run) {
    element.innerHTML = `
      <style>
        #points-runs { overflow: hidden auto; max-height: ${RUNS_SHOWN * RUN_ROW_PX}px; background: var(--panel); }
        #points-runs > div { position: relative; }
        #points-runs button { position: absolute; left: 0; right: 0; height: ${RUN_ROW_PX - 2}px; padding: 0 10px;
          display: grid; grid-template-columns: minmax(0, 1fr) auto; gap: 8px; align-items: center; text-align: left; }
        #points-runs button span:first-child { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
        #points-runs button.active { border-color: var(--accent); color: var(--accent); }
        .points-runs-head { display: flex; justify-content: space-between; }
      </style>
      <div class="points-runs-head"><span class="control-name">Marked runs</span>
        <span id="points-runs-count" title=", and . step through every run"></span></div>
      <div id="points-runs"><div></div></div>`;
    this.element = element;
    this.list = element.querySelector("#points-runs");
    this.count = element.querySelector("#points-runs-count");
    this.rows = this.list.firstElementChild;
    this.listed = null;
    this.scrolledTo = -1;
    this.laid = new Map();
    this.list.addEventListener("scroll", () => this.lay());
    this.list.addEventListener("click", (event) => {
      const button = event.target.closest("button");
      if (button === null) return;
      run(Number(button.dataset.run));
    });
  }

  show(runs, runRow, currentRun) {
    this.element.hidden = runs.length === 0;
    this.count.textContent = counted(runs.length, "run");
    this.listed = { runs, runRow, currentRun };
    this.lay();
  }

  lay() {
    const { runs, runRow, currentRun } = this.listed;
    const list = this.list;
    this.rows.style.height = `${runs.length * RUN_ROW_PX}px`;
    if (currentRun !== this.scrolledTo && currentRun >= 0) {
      const top = currentRun * RUN_ROW_PX;
      if (top < list.scrollTop) list.scrollTop = top;
      else if (top + RUN_ROW_PX > list.scrollTop + list.clientHeight)
        list.scrollTop = top + RUN_ROW_PX - list.clientHeight;
    }
    this.scrolledTo = currentRun;
    const first = Math.max(0, Math.floor(list.scrollTop / RUN_ROW_PX) - RUNS_BEYOND);
    const end = Math.min(runs.length, first + RUNS_SHOWN + 2 * RUNS_BEYOND + 1);
    for (const [index, button] of this.laid)
      if (index < first || index >= end) {
        button.remove();
        this.laid.delete(index);
      }
    for (let index = first; index < end; index++) {
      const { label, count } = runRow(runs[index]);
      let button = this.laid.get(index);
      if (button === undefined) {
        button = Object.assign(document.createElement("button"), { type: "button" });
        button.append(document.createElement("span"), document.createElement("span"));
        button.dataset.run = String(index);
        button.style.top = `${index * RUN_ROW_PX}px`;
        this.rows.append(button);
        this.laid.set(index, button);
      }
      if (button.title !== label) button.title = button.firstChild.textContent = label;
      if (button.lastChild.textContent !== count) button.lastChild.textContent = count;
      button.classList.toggle("active", index === currentRun);
    }
  }
}
