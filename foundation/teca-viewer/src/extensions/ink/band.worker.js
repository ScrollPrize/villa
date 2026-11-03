import { boxMesh, chunkBand, prepared } from "./band.js";

let surface = null;
let image = null;

self.onmessage = ({ data }) => {
  if (data.surface !== undefined) {
    surface = prepared(data.surface);
    image = data.image;
  } else if (data.scale !== undefined) {
    const data_DHW = chunkBand(surface, image, data.scale, data.chunk_ZYX, data.steps);
    self.postMessage(data_DHW, data_DHW === null ? [] : [data_DHW.buffer]);
  } else {
    const mesh = boxMesh(surface, data.low_ZYX, data.high_ZYX);
    self.postMessage(mesh, [mesh.positions_xyz_N3.buffer, mesh.uvs_N2.buffer, mesh.triangles_T3.buffer]);
  }
};
