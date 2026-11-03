import { HELD_SLICE_STEP_INCREMENT, ROTATION_STEP_DEGREES } from "../config.js";
import { surfaceControlsShown } from "./surface-controls.js";

const FOCUSABLE_CONTROLS = "input, select, textarea, button, summary, [role='button'], [role='listbox']";
const TYPED_CONTROLS = "input, select, textarea, [role='listbox']";
const TYPED_FIELDS =
  "input:is(:not([type]), [type='text'], [type='number'], [type='search']), textarea, [role='listbox']";
// The keys whose keydown a control acts on though blurred at it: a select steps its choice or opens its list, a slider
// or a radio steps, and a button, a checkbox or a summary begins a press. The rest, a select's first letters among
// them, it reads at their keypress, which goes to the page once the control is blurred.
const CONTROL_KEYS = [
  "ArrowUp",
  "ArrowDown",
  "ArrowLeft",
  "ArrowRight",
  "PageUp",
  "PageDown",
  "Home",
  "End",
  "F4",
  " ",
  "Enter",
];

function bindingMatches(binding, event, tool) {
  // A remapped key reports the physical code it was sent from while carrying the logical key,
  // so named keys and Space also match on `key`. Letters, digits, and punctuation keep their
  // physical positions, since their codes never equal a `key` value.
  const logicalCode = event.key === " " ? "Space" : event.key;
  if (binding.code !== event.code && binding.code !== logicalCode) return false;
  const command = event.ctrlKey || event.metaKey;
  if (binding.shift !== undefined && binding.shift !== event.shiftKey) return false;
  if (binding.command !== undefined && binding.command !== command) return false;
  if (binding.alt !== undefined && binding.alt !== event.altKey) return false;
  if (binding.tool !== undefined && binding.tool !== tool) return false;
  return binding.when === undefined || binding.when();
}

export function pressedBinding(bindings, event, tool) {
  const binding = bindings.find((binding) => bindingMatches(binding, event, tool));
  return { binding, runs: binding !== undefined && (!event.repeat || binding.repeats === true) };
}

export function createKeyboardControls({
  state,
  surfaceController,
  view,
  adjustHeldSliceStep,
  resetRotation,
  restoreTemporaryCamera,
  saveTemporaryCamera,
  setActiveViewport,
  setPlane,
  toggleActiveFullscreen,
  toggleCrosshair,
  toggleNormalLock,
  toggleAlone,
  cancelPointerInteraction,
  focusAtPointer,
  holdPan,
  releasePan,
  startHeldSlice,
  startHeldZoom,
  startHeldRotation,
  stepOneSlice,
  stopHeldSlice,
  stopHeldZoom,
  stopHeldRotation,
  modalIsOpen,
  toggleLeftPanel,
  toggleRightPanel,
  shortcuts,
}) {
  let pressedControl = null;
  const oneSlice = { shift: true, command: false, alt: false, repeats: true, when: volumeLoaded };
  const KEY_BINDINGS = [
    {
      description: "Reset Surface framing",
      code: "Home",
      shift: false,
      command: false,
      alt: false,
      when: () => state.activeViewport === "surface" && surfaceController.active(),
      run: () => surfaceController.resetFraming(),
    },
    {
      description: "Reset Surface normal offset",
      code: "Home",
      shift: true,
      command: false,
      alt: false,
      when: () => state.activeViewport === "surface" && surfaceController.active(),
      run: () => surfaceController.setNormalOffset(0),
    },
    {
      description: "Reset slab camera",
      code: "Home",
      shift: false,
      command: false,
      alt: true,
      when: () => state.activeViewport === "surface" && surfaceController.active(),
      run: () => surfaceController.resetCamera(),
    },
    {
      description: "Lock or unlock the Surface normal",
      code: "KeyL",
      shift: true,
      command: false,
      alt: false,
      when: () => surfaceControlsShown(state),
      run: toggleNormalLock,
    },
    {
      description: "Open Surface / return to CT panes",
      code: "KeyU",
      shift: false,
      command: false,
      alt: false,
      run: () => {
        void surfaceController
          .openMode(surfaceController.active() ? "planes" : "surface")
          .catch((error) => surfaceController.handleOpenError(error));
      },
    },
    {
      description: "Toggle left panel",
      code: "KeyK",
      shift: false,
      command: false,
      alt: false,
      run: toggleLeftPanel,
    },
    {
      description: "Toggle right panel",
      code: "KeyL",
      shift: false,
      command: false,
      alt: false,
      run: toggleRightPanel,
    },
    {
      description: "Move backward through depth",
      code: "KeyD",
      shift: false,
      command: false,
      alt: false,
      repeats: true,
      when: volumeLoaded,
      run: (event) => startHeldSlice(-1, event.repeat),
    },
    {
      description: "Move forward through depth",
      code: "KeyF",
      shift: false,
      command: false,
      alt: false,
      repeats: true,
      when: volumeLoaded,
      run: (event) => startHeldSlice(1, event.repeat),
    },
    { description: "Move one slice backward", code: "KeyD", ...oneSlice, run: () => stepOneSlice(-1) },
    { description: "Move one slice forward", code: "KeyF", ...oneSlice, run: () => stepOneSlice(1) },
    {
      description: "Focus XY pane",
      code: "Digit1",
      shift: false,
      command: false,
      alt: false,
      when: volumeLoaded,
      run: () => setPlane("xy"),
    },
    {
      description: "Focus ZY pane",
      code: "Digit2",
      shift: false,
      command: false,
      alt: false,
      when: volumeLoaded,
      run: () => setPlane("zy"),
    },
    {
      description: "Focus XZ pane",
      code: "Digit3",
      shift: false,
      command: false,
      alt: false,
      when: volumeLoaded,
      run: () => setPlane("xz"),
    },
    {
      description: "Focus 3D overview",
      code: "Digit4",
      shift: false,
      command: false,
      alt: false,
      when: volumeLoaded,
      run: focusThreePane,
    },
    {
      description: "Center all views at the mouse",
      code: "KeyR",
      shift: false,
      command: false,
      alt: false,
      when: volumeLoaded,
      run: focusAtPointer,
    },
    {
      description: "Reset CT plane rotation / Surface normal offset",
      code: "Digit0",
      shift: false,
      command: false,
      alt: false,
      when: volumeLoaded,
      run: () => {
        cancelPointerInteraction();
        if (state.activeViewport === "surface" && surfaceController.active()) surfaceController.setNormalOffset(0);
        else resetRotation();
      },
    },
    {
      description: "Toggle crosshair",
      code: "KeyV",
      shift: false,
      command: false,
      alt: false,
      run: toggleCrosshair,
    },
    {
      description: "Save temporary camera",
      code: "KeyY",
      shift: false,
      command: false,
      alt: false,
      when: volumeLoaded,
      run: saveTemporaryCamera,
    },
    {
      description: "Restore temporary camera",
      code: "KeyY",
      shift: true,
      command: false,
      alt: false,
      when: volumeLoaded,
      run: restoreTemporaryCamera,
    },
    {
      description: "Maximize or restore active pane",
      code: "KeyZ",
      shift: false,
      command: false,
      alt: false,
      when: volumeLoaded,
      run: toggleActiveFullscreen,
    },
    {
      description: "Zoom in",
      code: "KeyI",
      shift: false,
      command: false,
      alt: false,
      repeats: true,
      when: volumeLoaded,
      run: (event) => startHeldZoom(1, event.repeat),
    },
    {
      description: "Zoom out",
      code: "KeyO",
      shift: false,
      command: false,
      alt: false,
      repeats: true,
      when: volumeLoaded,
      run: (event) => startHeldZoom(-1, event.repeat),
    },
    {
      description: "Slower depth navigation",
      code: "KeyG",
      shift: false,
      command: false,
      alt: false,
      repeats: true,
      when: volumeLoaded,
      run: () => adjustHeldSliceStep(-HELD_SLICE_STEP_INCREMENT),
    },
    {
      description: "Faster depth navigation",
      code: "KeyH",
      shift: false,
      command: false,
      alt: false,
      repeats: true,
      when: volumeLoaded,
      run: () => adjustHeldSliceStep(HELD_SLICE_STEP_INCREMENT),
    },
    {
      description: "Rotate active view left",
      code: "ArrowLeft",
      repeats: true,
      when: volumeLoaded,
      run: (event) => startHeldRotation(event.code, "up", -ROTATION_STEP_DEGREES, event.repeat),
    },
    {
      description: "Rotate active view right",
      code: "ArrowRight",
      repeats: true,
      when: volumeLoaded,
      run: (event) => startHeldRotation(event.code, "up", ROTATION_STEP_DEGREES, event.repeat),
    },
    {
      description: "Rotate active view up",
      code: "ArrowUp",
      repeats: true,
      when: volumeLoaded,
      run: (event) => startHeldRotation(event.code, "right", -ROTATION_STEP_DEGREES, event.repeat),
    },
    {
      description: "Rotate active view down",
      code: "ArrowDown",
      repeats: true,
      when: volumeLoaded,
      run: (event) => startHeldRotation(event.code, "right", ROTATION_STEP_DEGREES, event.repeat),
    },
    {
      description: "Show the CT alone, or with its layers and images again",
      code: "Space",
      shift: false,
      command: false,
      alt: false,
      when: () => state.volume !== null && state.tool === null,
      run: () => toggleAlone("ct"),
    },
    {
      description: "Show the layers and images alone, or with the CT again",
      code: "Space",
      shift: false,
      command: true,
      alt: false,
      when: volumeLoaded,
      run: () => toggleAlone("layers"),
    },
    {
      description: "Pan while a tool is set; a tap with no click shows the CT alone, or both again",
      code: "Space",
      repeats: true,
      when: () => state.tool !== null,
      run: holdPan,
    },
  ];

  function addKeys(bindings) {
    for (const binding of bindings) {
      const taken = KEY_BINDINGS.find(
        (other) =>
          other.code === binding.code &&
          ["shift", "command", "alt"].every(
            (modifier) =>
              other[modifier] === undefined || binding[modifier] === undefined || other[modifier] === binding[modifier],
          ) &&
          (other.tool === undefined || binding.tool === undefined || other.tool === binding.tool),
      );
      if (taken !== undefined)
        throw new Error(`Key ${binding.code} of "${binding.description}" is taken by "${taken.description}".`);
      KEY_BINDINGS.push(binding);
    }
  }

  function initKeyboard() {
    for (const binding of KEY_BINDINGS) {
      const key = document.createElement("dt");
      key.textContent = [
        binding.command ? "Ctrl/⌘" : null,
        binding.shift ? "Shift" : null,
        binding.alt ? "Alt" : null,
        binding.code.replace(/^(Key|Digit)/, ""),
      ]
        .filter(Boolean)
        .join("+");
      const description = document.createElement("dd");
      description.textContent = binding.description;
      const held = document.createElement("dd");
      held.textContent = binding.repeats === true ? "while held" : "";
      shortcuts.append(key, description, held);
    }
    window.addEventListener(
      "pointerdown",
      (event) => {
        const pressed = event.target instanceof Element ? event.target : null;
        const control = pressed?.closest(FOCUSABLE_CONTROLS) ?? pressed?.closest("label")?.control ?? null;
        pressedControl = control?.matches(TYPED_FIELDS) ? null : control;
      },
      { capture: true },
    );
    window.addEventListener("focusin", (event) => {
      if (event.target !== pressedControl) pressedControl = null;
    });
    window.addEventListener("keydown", onKeyDown);
    window.addEventListener("keyup", onKeyUp);
    // A key held while the window loses focus sends no keyup.
    window.addEventListener("blur", () => {
      if (state.heldSliceDirection !== 0) stopHeldSlice();
      if (state.heldZoomDirection !== 0) stopHeldZoom();
      if (state.heldRotation !== null) stopHeldRotation();
      releasePan();
    });
  }

  function onKeyDown(event) {
    if (event.isComposing) return;
    if (modalIsOpen()) return;
    let target = event.target;
    if (target === pressedControl) {
      pressedControl = null;
      target.blur();
      target = null;
      if (CONTROL_KEYS.includes(event.key)) event.preventDefault();
    }
    if (event.code === "Escape" || event.key === "Escape") {
      const focused = document.activeElement;
      if (focused?.matches(FOCUSABLE_CONTROLS) || focused?.isContentEditable) {
        event.preventDefault();
        focused.blur();
        return;
      }
    }
    if (target instanceof Element && (target.matches(TYPED_CONTROLS) || target.isContentEditable)) return;
    if (event.code === "Tab" && target instanceof Element && target.matches(FOCUSABLE_CONTROLS)) return;
    const { binding, runs } = pressedBinding(KEY_BINDINGS, event, state.tool);
    if (binding === undefined) return;
    event.preventDefault();
    if (runs) binding.run(event);
  }

  // Released by the physical key that started the hold, whatever the layout or Shift make of it.
  function onKeyUp(event) {
    if ((event.code === "Space" || event.key === " ") && releasePan()) toggleAlone("ct");
    if (
      (event.code === "KeyD" && state.heldSliceDirection < 0) ||
      (event.code === "KeyF" && state.heldSliceDirection > 0)
    )
      stopHeldSlice();
    if (
      (event.code === "KeyI" && state.heldZoomDirection > 0) ||
      (event.code === "KeyO" && state.heldZoomDirection < 0)
    )
      stopHeldZoom();
    if (event.code === state.heldRotation?.code) stopHeldRotation();
  }

  function volumeLoaded() {
    return state.volume !== null;
  }

  function focusThreePane() {
    if (setActiveViewport("3d", { redrawOnly: true })) {
      view.getCanvas("3d").focus({ preventScroll: true });
    }
  }

  return { initKeyboard, addKeys };
}
