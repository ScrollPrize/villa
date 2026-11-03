import { SURFACE_DOCK_PANES } from "../surface/layout.js";

export class SurfaceLayout {
  constructor(grid, panes) {
    this.grid = grid;
    this.planes = panes.filter((pane) => pane.mode === "plane" || pane.mode === "3d");
    this.surface = panes.find((pane) => pane.name === "surface");
    this.strip = document.createElement("div");
    this.strip.className = "surface-plane-strip";
    this.divider = document.createElement("div");
    this.divider.className = "surface-layout-divider";
    this.divider.dataset.split = "main";
    this.dividers = this.planes.slice(1).map((_, index) => {
      const divider = document.createElement("div");
      divider.className = "surface-layout-divider";
      divider.dataset.split = String(index);
      return divider;
    });
    for (const divider of [this.divider, ...this.dividers]) {
      divider.setAttribute("role", "separator");
      divider.tabIndex = 0;
      divider.setAttribute(
        "aria-label",
        divider === this.divider ? "Resize Surface and planes" : "Resize adjacent planes",
      );
    }
    this.signature = null;
  }

  apply(layout, active, primary = "surface") {
    const mode = active ? layout.mode : "solo";
    const tiled = mode !== "solo" && layout.maximized === null;
    this.grid.classList.toggle("surface-tiling", tiled);
    this.grid.dataset.surfaceLayout = mode;
    this.grid.dataset.contextPrimary = primary;
    this.divider.setAttribute(
      "aria-label",
      primary === "surface" ? "Resize Surface and planes" : "Resize 3D and CT context",
    );
    const signature = `${primary}:${mode}:${layout.planes.join(",")}:${layout.maximized}:${active}`;
    if (signature !== this.signature) {
      this.signature = signature;
      this.divider.remove();
      this.strip.remove();
      for (const divider of this.dividers) divider.remove();
      this.grid.prepend(...this.planes.map((pane) => pane.element));
      if (tiled) {
        const children = [];
        for (const [index, name] of layout.planes.entries()) {
          if (index > 0) children.push(this.dividers[index - 1]);
          children.push(this.planes.find((pane) => pane.name === name).element);
        }
        this.strip.replaceChildren(...children);
        this.grid.append(this.divider, this.strip);
      }
      this.grid.dataset.surfaceMaximized = active ? (layout.maximized ?? "") : "";
    }
    for (const pane of this.planes) {
      pane.element.hidden =
        active &&
        !(layout.maximized === null && (pane.name === primary || (tiled && layout.planes.includes(pane.name))));
    }
    if (this.surface !== undefined)
      this.surface.setVisible(
        active && primary === "surface" && (layout.maximized === null || layout.maximized === "surface"),
      );
    if (active && layout.maximized !== null && layout.maximized !== "surface") {
      this.planes.find((pane) => pane.name === layout.maximized).element.hidden = false;
    }
    if (!tiled) {
      this.grid.style.removeProperty("grid-template-columns");
      this.grid.style.removeProperty("grid-template-rows");
      return;
    }
    const main = `${layout.split}fr 6px ${1 - layout.split}fr`;
    this.grid.style.gridTemplateColumns = mode === "right" ? main : "minmax(0, 1fr)";
    this.grid.style.gridTemplateRows = mode === "below" ? main : "minmax(0, 1fr)";
    const sizes = layout.planes
      .map((plane) => `minmax(0, ${layout.weights[SURFACE_DOCK_PANES.indexOf(plane)]}fr)`)
      .join(" 6px ");
    this.strip.style.gridTemplateColumns = mode === "below" ? sizes : "minmax(0, 1fr)";
    this.strip.style.gridTemplateRows = mode === "right" ? sizes : "minmax(0, 1fr)";
    for (const divider of [this.divider, ...this.dividers]) {
      const vertical = divider === this.divider ? mode === "right" : mode === "below";
      divider.setAttribute("aria-orientation", vertical ? "vertical" : "horizontal");
    }
    this.divider.setAttribute("aria-valuenow", String(Math.round(layout.split * 100)));
    this.divider.setAttribute("aria-valuemin", "20");
    this.divider.setAttribute("aria-valuemax", "85");
  }
}
