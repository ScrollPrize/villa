import { planPlaneDemand } from "./plane-demand.js";

self.onmessage = ({ data }) => {
  try {
    self.postMessage({ id: data.id, data: planPlaneDemand(data) });
  } catch (error) {
    self.postMessage({ id: data.id, error: error.message });
  }
};
