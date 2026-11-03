export function createPopovers(document) {
  let current = null;

  function close() {
    if (current === null) return;
    current.popover.hidden = true;
    current.button.setAttribute("aria-expanded", "false");
    current = null;
  }

  document.addEventListener("click", (event) => {
    if (current === null) return;
    if (current.popover.contains(event.target) || current.button.contains(event.target)) return;
    // A click whose target was re-rendered away during dispatch is not an outside click.
    if (!event.target.isConnected) return;
    close();
  });
  document.addEventListener("keydown", (event) => {
    if (event.key !== "Escape" || current === null) return;
    event.preventDefault();
    event.stopPropagation();
    close();
    document.activeElement?.blur();
  });

  function wirePopover(button, popover) {
    const open = () => {
      if (current !== null && current.popover !== popover) close();
      popover.hidden = false;
      button.setAttribute("aria-expanded", "true");
      current = { button, popover };
    };
    button.addEventListener("click", () => {
      if (current !== null && current.popover === popover) close();
      else open();
    });
    return { open, close: () => current !== null && current.popover === popover && close() };
  }

  return wirePopover;
}
