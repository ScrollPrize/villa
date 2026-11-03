export function createViewerState(session) {
  return {
    ...session,
    workGeneration: 0,
    contentGeneration: 0,
    volume: null,
    crosshairVisible: true,
    alone: null,
    temporaryCamera: null,
    renderToken: 0,
    dirtyPlanes: null,
    lastInteractionDirtyPlanes: null,
    heldSliceDirection: 0,
    heldZoomDirection: 0,
    heldRotation: null,
    pointerDrag: null,
    lastPointerClient: null,
    tool: null,
    panHeld: false,
  };
}
