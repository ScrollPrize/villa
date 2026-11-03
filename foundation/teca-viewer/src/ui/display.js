import { DISPLAY_GAMMA_RANGE } from "../config.js";
import { clamp } from "../math.js";
import { formatControlNumber } from "./readouts.js";

export function createDisplay({ state, elements, view, publishSession, queueOrthogonalRedraw, setStatus }) {
  function displaySettings() {
    const black = clamp(state.displayBlack, 0, 254);
    const white = clamp(Math.max(state.displayWhite, black + 1), 1, 255);
    return {
      black: black / 255,
      white: white / 255,
      gamma: clamp(state.displayGamma, ...DISPLAY_GAMMA_RANGE),
    };
  }

  function syncDisplayInputs() {
    elements.displayBlack.value = state.displayBlack;
    elements.displayBlackNumber.value = state.displayBlack;
    elements.displayWhite.value = state.displayWhite;
    elements.displayWhiteNumber.value = state.displayWhite;
    elements.displayGamma.value = state.displayGamma;
    elements.displayGammaNumber.value = formatControlNumber(state.displayGamma);
  }

  function setDisplayControlValue(name, value) {
    const next = {
      black: state.displayBlack,
      white: state.displayWhite,
      gamma: state.displayGamma,
    };
    if (name === "black") {
      next.black = Math.round(value);
    } else if (name === "white") {
      next.white = Math.round(value);
    } else {
      next.gamma = value;
    }
    if (
      !Number.isFinite(next.black) ||
      !Number.isFinite(next.white) ||
      !Number.isFinite(next.gamma) ||
      next.black < 0 ||
      next.black > 254 ||
      next.white < 1 ||
      next.white > 255 ||
      next.white <= next.black ||
      next.gamma < DISPLAY_GAMMA_RANGE[0] ||
      next.gamma > DISPLAY_GAMMA_RANGE[1]
    ) {
      setStatus("Display values are invalid: white must exceed black and all values must remain in range.");
      syncDisplayInputs();
      return;
    }
    state.displayBlack = next.black;
    state.displayWhite = next.white;
    state.displayGamma = next.gamma;
    syncDisplayInputs();
    view.setDisplaySettings(displaySettings());
    publishSession();
    queueOrthogonalRedraw();
  }

  function showAlone(alone) {
    state.alone = alone;
    view.showAlone(alone);
    queueOrthogonalRedraw();
  }

  function toggleAlone(alone) {
    if (alone === "layers" && state.alone !== "layers" && !view.hasLayers()) {
      setStatus("No layer or image is drawn over the CT, so Ctrl+Space leaves it shown.");
      return;
    }
    showAlone(state.alone === alone ? null : alone);
    if (state.alone === "ct") setStatus("The CT alone, its layers and images hidden: Space shows them again.");
    else if (state.alone === "layers")
      setStatus("The layers and images alone, the CT hidden: Ctrl+Space shows it again.");
    else
      setStatus("The CT with its layers and images: Space shows the CT alone, Ctrl+Space the layers and images alone.");
  }

  return { displaySettings, syncDisplayInputs, setDisplayControlValue, showAlone, toggleAlone };
}
