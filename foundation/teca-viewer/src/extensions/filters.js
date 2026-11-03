import { KINDS, stagePasses } from "./filters/passes.js";

export function savedStages(saved) {
  if (!Array.isArray(saved)) throw new Error(`Saved filters must be an array: ${JSON.stringify(saved)}.`);
  for (const stage of saved) {
    const parameter = KINDS[stage?.kind]?.parameter;
    if (
      parameter === undefined ||
      Object.keys(stage).sort().join() !== "enabled,kind,value" ||
      typeof stage.enabled !== "boolean" ||
      (parameter === null ? stage.value !== null : !(stage.value >= parameter.min && stage.value <= parameter.max))
    )
      throw new Error(`Saved filter is malformed: ${JSON.stringify(stage)}.`);
  }
  return saved;
}

export default function createFilters(viewer, entry) {
  const panel = document.createElement("section");
  panel.id = "filters-section";
  panel.className = "card source-dependent";
  panel.setAttribute("aria-label", "Filters");
  panel.innerHTML = `<style>.filters-value { flex: none; width: 3.5em; }</style>
    <h2>Filters</h2>
    <div id="filters-items" class="overlay-items"></div>
    <div class="button-row">
      <select id="filters-kind" aria-label="Filter">
        ${Object.entries(KINDS)
          .map(([kind, { name }]) => `<option value="${kind}">${name}</option>`)
          .join("")}
      </select>
      <button id="filters-add" type="button">Add</button>
    </div>`;
  const items = panel.querySelector("#filters-items");
  const kindSelect = panel.querySelector("#filters-kind");
  const addButton = panel.querySelector("#filters-add");

  let stages = [];
  try {
    stages = savedStages(entry.saved ?? []).map((saved) => ({ ...saved, passes: stagePasses(saved) }));
  } catch (error) {
    viewer.setStatus(`The link's filters were not used: ${error.message}`);
  }

  function changed() {
    entry.save(stages.map(({ kind, enabled, value }) => ({ kind, enabled, value })));
    const used = stages.filter((stage) => stage.enabled);
    viewer.filterCT(
      used.flatMap((stage) => stage.passes),
      used.reduce((reach, stage) => reach + KINDS[stage.kind].reach(stage.value), 0),
    );
    items.replaceChildren(
      ...stages.map((stage, index) => {
        const row = document.createElement("div");
        row.className = "layer-row";
        const enabled = Object.assign(document.createElement("input"), { type: "checkbox", checked: stage.enabled });
        enabled.setAttribute("aria-label", `Use ${KINDS[stage.kind].name}`);
        enabled.addEventListener("change", () => {
          stage.enabled = enabled.checked;
          changed();
        });
        const name = Object.assign(document.createElement("span"), {
          className: "layer-row-name",
          textContent: KINDS[stage.kind].name,
        });
        row.append(enabled, name);
        const parameter = KINDS[stage.kind].parameter;
        if (parameter !== null) {
          const input = Object.assign(document.createElement("input"), {
            type: "number",
            min: parameter.min,
            max: parameter.max,
            step: parameter.step,
            value: stage.value,
            className: "filters-value",
          });
          input.setAttribute("aria-label", `${KINDS[stage.kind].name} ${parameter.label}`);
          input.addEventListener("change", () => {
            if (input.valueAsNumber >= parameter.min && input.valueAsNumber <= parameter.max) {
              stage.value = input.valueAsNumber;
              for (const pass of stage.passes) pass.uniforms.value.value = stage.value;
            }
            changed();
          });
          row.append(input);
        }
        for (const [label, title, move] of [
          ["↑", "Earlier", -1],
          ["↓", "Later", 1],
          ["×", "Remove", 0],
        ]) {
          const button = Object.assign(document.createElement("button"), { type: "button", textContent: label, title });
          button.disabled = move !== 0 && !(index + move >= 0 && index + move < stages.length);
          button.addEventListener("click", () => {
            if (move === 0) for (const pass of stages.splice(index, 1)[0].passes) pass.dispose();
            else stages.splice(index + move, 0, stages.splice(index, 1)[0]);
            changed();
          });
          row.append(button);
        }
        return row;
      }),
    );
  }

  addButton.addEventListener("click", () => {
    const stage = { kind: kindSelect.value, enabled: true, value: KINDS[kindSelect.value].parameter?.value ?? null };
    stages.push({ ...stage, passes: stagePasses(stage) });
    changed();
  });

  changed();
  return { panel, place: "data" };
}
