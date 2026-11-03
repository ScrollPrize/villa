const ICON = `<svg viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linejoin="round"><path d="M6 3h8v14l-4-3.5L6 17z"/></svg>`;
const NUMBER = String.raw`(-?(?:\d+\.?\d*|\.\d+))`;
const POSITION = new RegExp(String.raw`^\[?\s*${NUMBER}\s*,\s*${NUMBER}\s*,\s*${NUMBER}\s*\]?(?:\s+-\s+(.*))?$`);
const ROW_PX = 40;
const ROWS_SHOWN = 4;

function positionText(point_ZYX) {
  return point_ZYX
    .toReversed()
    .map((value) => String(Number(value.toFixed(3))))
    .join(", ");
}

function inside(point_ZYX, shape_ZYX) {
  return point_ZYX.every((value, axis) => value >= 0 && value < shape_ZYX[axis]);
}

export function typedBookmarks(text, shape_ZYX, focus_ZYX) {
  const lines = text
    .split(/\r\n|\r|\n/)
    .map((line, index) => ({ number: index + 1, line: line.trim() }))
    .filter(({ line }) => line !== "");
  if (lines.length <= 1 && !POSITION.test(lines[0]?.line ?? ""))
    return [{ point_ZYX: focus_ZYX, label: lines[0]?.line ?? "" }];
  return lines.map(({ number, line }) => {
    const found = POSITION.exec(line);
    if (found === null) throw new Error(`Line ${number}, ${JSON.stringify(line)}, is not x, y, z.`);
    const point_ZYX = [Number(found[3]), Number(found[2]), Number(found[1])];
    if (!inside(point_ZYX, shape_ZYX))
      throw new Error(
        `Line ${number}, ${JSON.stringify(line)}, lies outside the CT, whose x, y and z are below ` +
          `${shape_ZYX.toReversed().join(", ")}.`,
      );
    return { point_ZYX, label: found[4] ?? "" };
  });
}

export function savedBookmarks(saved) {
  if (saved === null || typeof saved !== "object" || Array.isArray(saved))
    throw new Error(`Saved bookmarks are lists keyed by a CT's URL, not ${JSON.stringify(saved)}.`);
  for (const [url, bookmarks] of Object.entries(saved)) {
    if (!/^https?:\/\/\S+\/$/.test(url) || !Array.isArray(bookmarks))
      throw new Error(`Saved bookmarks are lists keyed by a CT's URL, not ${JSON.stringify({ [url]: bookmarks })}.`);
    for (const bookmark of bookmarks) {
      const { point_ZYX, label } = bookmark ?? {};
      if (
        Object.keys(bookmark ?? {})
          .sort()
          .join() !== "label,point_ZYX" ||
        !Array.isArray(point_ZYX) ||
        point_ZYX.length !== 3 ||
        !point_ZYX.every((value) => Number.isFinite(value) && value >= 0) ||
        typeof label !== "string" ||
        /[\r\n]/.test(label)
      )
        throw new Error(`Saved bookmark is malformed: ${JSON.stringify(bookmark)}.`);
    }
  }
  return saved;
}

export default function createBookmarks(viewer, entry) {
  const panel = document.createElement("section");
  panel.id = "bookmarks-section";
  panel.className = "card";
  panel.setAttribute("aria-label", "Bookmarks");
  panel.innerHTML = `<style>
      #bookmarks-section { display: flex; flex-direction: column; }
      #bookmarks-text { min-height: 44px; padding: 5px 8px; border: 1px solid;
        border-color: var(--bevel-dark) var(--bevel-light) var(--bevel-light) var(--bevel-dark);
        border-radius: var(--radius-sm); background: var(--field); color: var(--ink); font: inherit; resize: vertical; }
      #bookmarks-text:focus-visible { outline: 1px solid var(--accent); outline-offset: -1px; }
      #bookmarks-note:empty { display: none; }
      #bookmarks-list { flex: 1 1 0; min-height: ${ROWS_SHOWN * ROW_PX}px; overflow-y: auto; }
      .bookmarks-row { display: flex; gap: 2px; height: ${ROW_PX}px; padding-bottom: 2px; }
      .bookmarks-go { flex: 1 1 auto; min-width: 0; display: grid; align-content: center; padding: 0 8px;
        text-align: left; font-size: 12px; line-height: 1.3; }
      .bookmarks-go span { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
      .bookmarks-go span:empty { display: none; }
      .bookmarks-go span + span { color: var(--muted); font-size: 11px; font-variant-numeric: tabular-nums; }
      .bookmarks-go.active { border-color: var(--accent); color: var(--accent); }
    </style>
    <h2>Bookmarks</h2>
    <div class="overlay-form">
      <textarea id="bookmarks-text" rows="2" spellcheck="false" aria-label="Label, or positions x, y, z"
        placeholder="A label for the focus, or x, y, z - label, a line each"></textarea>
      <button id="bookmarks-add" type="button">Add</button>
    </div>
    <p id="bookmarks-note" class="readout-line" role="status"></p>
    <div id="bookmarks-list"></div>`;
  const text = panel.querySelector("#bookmarks-text");
  const addButton = panel.querySelector("#bookmarks-add");
  const note = panel.querySelector("#bookmarks-note");
  const list = panel.querySelector("#bookmarks-list");

  let saved = {};
  try {
    saved = savedBookmarks(entry.saved ?? {});
  } catch (error) {
    viewer.setStatus(`The link's bookmarks were not used: ${error.message}`);
  }
  let ct = null;
  let visited = null;

  function keep(bookmarks) {
    const next = { ...saved, [ct.rootUrl]: bookmarks };
    if (bookmarks.length === 0) delete next[ct.rootUrl];
    saved = next;
    return entry.save(next);
  }

  function describe() {
    text.disabled = ct === null;
    addButton.disabled = ct === null;
    list.replaceChildren(
      ...(ct === null ? [] : (saved[ct.rootUrl] ?? [])).map((bookmark) => {
        const position = positionText(bookmark.point_ZYX);
        const go = Object.assign(document.createElement("button"), {
          type: "button",
          className: "bookmarks-go",
          title: bookmark.label === "" ? position : `${bookmark.label}\n${position}`,
        });
        go.classList.toggle("active", bookmark === visited);
        go.append(
          Object.assign(document.createElement("span"), { textContent: bookmark.label }),
          Object.assign(document.createElement("span"), { textContent: position }),
        );
        go.addEventListener("click", () => {
          viewer.focus(bookmark.point_ZYX);
          visited = bookmark;
          describe();
        });
        const remove = Object.assign(document.createElement("button"), {
          type: "button",
          textContent: "×",
          title: "Delete",
        });
        remove.addEventListener("click", () => {
          keep(saved[ct.rootUrl].filter((other) => other !== bookmark));
          if (visited === bookmark) visited = null;
          note.textContent = "";
          describe();
        });
        const row = Object.assign(document.createElement("div"), { className: "bookmarks-row" });
        row.append(go, remove);
        return row;
      }),
    );
  }

  function add() {
    let added;
    try {
      added = typedBookmarks(text.value, ct.scales[0].shape_ZYX, viewer.focusPoint());
    } catch (error) {
      note.textContent = `${error.message} Nothing was added.`;
      return false;
    }
    const held = saved[ct.rootUrl] ?? [];
    if (!keep([...held, ...added])) {
      keep(held);
      const more = added.length === 1 ? "another bookmark" : `${added.length} more bookmarks`;
      note.textContent = `The link cannot hold ${more}. Nothing was added.`;
      return false;
    }
    text.value = "";
    note.textContent = "";
    describe();
    list.lastElementChild.scrollIntoView({ block: "nearest" });
    return true;
  }

  addButton.addEventListener("click", add);
  text.addEventListener("keydown", (event) => {
    if (event.key !== "Enter" || event.shiftKey || event.isComposing) return;
    event.preventDefault();
    if (add()) text.blur();
  });

  function volumeChanged(opened) {
    ct = opened;
    visited = null;
    note.textContent = "";
    const stray = (saved[ct.rootUrl] ?? []).find(({ point_ZYX }) => !inside(point_ZYX, ct.scales[0].shape_ZYX));
    if (stray !== undefined) {
      viewer.setStatus(
        `The link's bookmarks of this CT were not used: ${JSON.stringify(stray)} lies outside its ` +
          `${ct.scales[0].shape_ZYX.toReversed().join(" × ")} voxels, x by y by z.`,
      );
      saved = Object.fromEntries(Object.entries(saved).filter(([url]) => url !== ct.rootUrl));
    }
    describe();
  }

  describe();
  return { panel, place: { title: "Bookmarks", icon: ICON }, volumeChanged };
}
