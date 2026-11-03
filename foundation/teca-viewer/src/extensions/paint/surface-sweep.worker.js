import { meshSwept } from "./surface-sweep.js";

self.onmessage = ({ data }) => {
  const swept = meshSwept(data);
  self.postMessage(swept, [swept.runs_N4.buffer, swept.tiles_N5.buffer]);
};
