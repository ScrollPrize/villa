const STORAGE = {
  leftCollapsed: "teca.viewer.leftCollapsed",
  rightCollapsed: "teca.viewer.rightCollapsed",
};

export function createShellPanels({ window, document, state, elements, storage, publishSession, onLayoutChange }) {
  const { app, toggleLeft, toggleRight } = elements;
  const narrowViewport = window.matchMedia("(max-width: 820px)");

  function tabs(side) {
    return [...document.querySelectorAll(`.${side[0]}tab`)];
  }

  function syncToggleState() {
    const leftOpen = narrowViewport.matches
      ? app.dataset.narrowPanel === "left"
      : !app.classList.contains("left-collapsed");
    const rightOpen = narrowViewport.matches
      ? app.dataset.narrowPanel === "right"
      : !app.classList.contains("right-collapsed");
    toggleLeft.setAttribute("aria-expanded", String(leftOpen));
    toggleRight.setAttribute("aria-expanded", String(rightOpen));
  }

  function showPanel(side) {
    if (narrowViewport.matches) {
      app.dataset.narrowPanel = side;
    } else {
      const wasCollapsed = app.classList.contains(`${side}-collapsed`);
      app.classList.remove(`${side}-collapsed`);
      storage.setItem(STORAGE[`${side}Collapsed`], "false");
      if (wasCollapsed) onLayoutChange();
    }
    syncToggleState();
  }

  function togglePanel(side) {
    if (narrowViewport.matches) {
      app.dataset.narrowPanel = app.dataset.narrowPanel === side ? "none" : side;
    } else {
      const className = `${side}-collapsed`;
      const collapsed = !app.classList.contains(className);
      app.classList.toggle(className, collapsed);
      storage.setItem(STORAGE[`${side}Collapsed`], String(collapsed));
      onLayoutChange();
    }
    syncToggleState();
  }

  function activateTab(side, name, reveal) {
    const key = `${side[0]}tab`;
    const selected = tabs(side).find((tab) => tab.dataset[key] === name && !tab.hidden);
    if (selected === undefined) throw new Error(`Unknown ${side} panel tab: ${name}`);
    for (const tab of tabs(side)) {
      const active = tab === selected;
      tab.classList.toggle("active", active);
      tab.setAttribute("aria-selected", String(active));
      tab.tabIndex = active ? 0 : -1;
    }
    for (const panel of document.querySelectorAll(`.${side[0]}panel`)) {
      panel.hidden = panel.dataset[`${side[0]}panel`] !== name;
    }
    state.tabs[side] = name;
    if (reveal) {
      showPanel(side);
      selected.scrollIntoView({ block: "nearest", inline: "nearest" });
    }
  }

  function moveTabFocus(event, side, current) {
    if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) return;
    const available = tabs(side).filter((tab) => !tab.hidden && !tab.disabled);
    let index = available.indexOf(current);
    if (event.key === "Home") index = 0;
    else if (event.key === "End") index = available.length - 1;
    else index = (index + (event.key === "ArrowRight" ? 1 : -1) + available.length) % available.length;
    event.preventDefault();
    event.stopPropagation();
    available[index].click();
    available[index].focus();
  }

  function syncViewportMode() {
    if (narrowViewport.matches) app.dataset.narrowPanel = "none";
    else delete app.dataset.narrowPanel;
    syncToggleState();
  }

  for (const side of ["left", "right"]) {
    if (storage.getItem(STORAGE[`${side}Collapsed`]) === "true") app.classList.add(`${side}-collapsed`);
    const key = `${side[0]}tab`;
    const selected =
      tabs(side).find((tab) => tab.dataset[key] === state.tabs[side] && !tab.hidden) ??
      tabs(side).find((tab) => !tab.hidden);
    if (selected !== undefined) activateTab(side, selected.dataset[key], false);
    for (const tab of tabs(side)) {
      tab.addEventListener("click", () => {
        activateTab(side, tab.dataset[key], true);
        if (state.volume !== null) publishSession();
      });
      tab.addEventListener("keydown", (event) => moveTabFocus(event, side, tab));
    }
  }
  const toggleLeftPanel = () => togglePanel("left");
  const toggleRightPanel = () => togglePanel("right");
  toggleLeft.addEventListener("click", toggleLeftPanel);
  toggleRight.addEventListener("click", toggleRightPanel);
  narrowViewport.addEventListener("change", syncViewportMode);
  syncViewportMode();
  if (
    !narrowViewport.matches &&
    (app.classList.contains("left-collapsed") || app.classList.contains("right-collapsed"))
  ) {
    onLayoutChange();
  }
  return { toggleLeftPanel, toggleRightPanel };
}
