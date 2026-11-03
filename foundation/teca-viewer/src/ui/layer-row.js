export function wireLayerRowMenu(wirePopover, button, popover) {
  const menu = wirePopover(button, popover);
  button.addEventListener("click", () => {
    if (!popover.hidden) popover.scrollIntoView({ block: "nearest" });
  });
  return menu;
}

export function layerRowMenu(wirePopover, row, name) {
  const button = document.createElement("button");
  button.type = "button";
  button.className = "icon-btn layer-row-menu";
  button.textContent = "⋯";
  button.setAttribute("aria-label", `Menu for ${name}`);
  button.setAttribute("aria-haspopup", "dialog");
  button.setAttribute("aria-expanded", "false");
  const popover = document.createElement("div");
  popover.className = "popover layer-row-popover";
  popover.setAttribute("role", "dialog");
  popover.setAttribute("aria-label", `${name} settings`);
  popover.hidden = true;
  row.append(button, popover);
  return { popover, ...wireLayerRowMenu(wirePopover, button, popover) };
}
