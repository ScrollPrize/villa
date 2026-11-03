import { planSurfaceDemand } from "./planning.js";

self.onmessage = ({ data }) => {
  try {
    self.postMessage({ id: data.id, data: planSurfaceDemand(data) });
  } catch (error) {
    self.postMessage({ id: data.id, error: error.message });
  }
};
