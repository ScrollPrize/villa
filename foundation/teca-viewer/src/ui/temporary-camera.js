export function createTemporaryCamera({ state, surfaceController, restorePose, setStatus }) {
  function saveTemporaryCamera() {
    if (state.volume === null || state.activeSource === null) return;
    state.temporaryCamera = {
      source: state.activeSource,
      center_ZYX: [...state.center_ZYX],
      zoom: state.zoom,
      plane: state.plane,
      activeViewport: state.activeViewport,
      fullscreenViewport: state.fullscreenViewport,
      rotationMode: state.rotationMode,
      rotation: [...state.rotation],
      paneRotations: structuredClone(state.paneRotations),
      cameraOrientation: [...state.cameraOrientation],
      surfaceView: structuredClone(state.surfaceView),
    };
    setStatus("Temporary camera saved.");
  }

  function restoreTemporaryCamera() {
    const saved = state.temporaryCamera;
    if (saved === null) {
      setStatus("No temporary camera has been saved.");
      return;
    }
    if (saved.source !== state.activeSource) {
      setStatus(`Temporary camera belongs to source ${saved.sourceId}.`);
      return;
    }
    const restoringSurface = saved.fullscreenViewport === "surface";
    if (surfaceController.active() && !restoringSurface) surfaceController.close();
    if (restoringSurface) surfaceController.restoreView(saved.surfaceView);
    restorePose({
      ...saved,
      activeViewport: restoringSurface ? saved.plane : saved.activeViewport,
      fullscreenViewport: restoringSurface ? null : saved.fullscreenViewport,
    });
    if (restoringSurface) {
      void surfaceController
        .openPath(state.surfaceView.path)
        .then(() => setStatus("Temporary Surface camera restored."))
        .catch((error) => {
          if (error.name === "AbortError") return;
          surfaceController.close();
          setStatus(error.message);
        });
    } else {
      setStatus("Temporary camera restored.");
    }
  }

  return { saveTemporaryCamera, restoreTemporaryCamera };
}
