function focusableElements(backdrop) {
  return [
    ...backdrop.querySelectorAll(
      "button:not([disabled]), input:not([disabled]), select:not([disabled]), textarea:not([disabled]), summary, [tabindex]:not([tabindex='-1'])",
    ),
  ].filter(
    (element) =>
      element.getClientRects().length > 0 &&
      // Closed details can retain layout boxes for their hidden content.
      element.closest("details:not([open]) > :not(summary)") === null,
  );
}

export function createShellModals({ document, elements, onSettingsOpen }) {
  let activeBackdrop = null;
  let returnFocus = null;

  function openModal(backdrop, trigger) {
    if (activeBackdrop !== null) closeModal(activeBackdrop, false);
    returnFocus = trigger;
    activeBackdrop = backdrop;
    backdrop.hidden = false;
    const initialFocus = backdrop.querySelector("[data-modal-focus]") ?? focusableElements(backdrop)[0];
    initialFocus.focus();
  }

  function closeModal(backdrop, restoreFocus = true) {
    if (backdrop.hidden) return;
    backdrop.hidden = true;
    if (activeBackdrop === backdrop) activeBackdrop = null;
    if (restoreFocus && returnFocus?.isConnected) returnFocus.focus();
    returnFocus = null;
  }

  for (const button of elements.openSourceButtons) {
    button.addEventListener("click", (event) => openModal(elements.openSourceModal, event.currentTarget));
  }
  elements.openShortcuts.addEventListener("click", (event) => {
    openModal(elements.shortcutsModal, event.currentTarget);
  });
  elements.openSettings.addEventListener("click", (event) => {
    openModal(elements.settingsModal, event.currentTarget);
    onSettingsOpen();
  });
  for (const backdrop of document.querySelectorAll(".modal-backdrop")) {
    backdrop.addEventListener("click", (event) => {
      if (event.target === backdrop || event.target.closest("[data-close-modal]") !== null) {
        closeModal(backdrop);
      }
    });
  }
  elements.form.addEventListener("submit", () => closeModal(elements.openSourceModal));
  document.addEventListener(
    "keydown",
    (event) => {
      if (activeBackdrop === null) return;
      if (event.key === "Escape") {
        event.preventDefault();
        event.stopPropagation();
        document.activeElement?.blur();
        closeModal(activeBackdrop, false);
        return;
      }
      if (event.key !== "Tab") return;
      const focusable = focusableElements(activeBackdrop);
      if (focusable.length === 0) {
        event.preventDefault();
        return;
      }
      const first = focusable[0];
      const last = focusable[focusable.length - 1];
      if (event.shiftKey && document.activeElement === first) {
        event.preventDefault();
        last.focus();
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault();
        first.focus();
      }
    },
    true,
  );

  return { modalIsOpen: () => activeBackdrop !== null, closeModal };
}
