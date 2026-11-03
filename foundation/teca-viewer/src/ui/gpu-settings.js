import { GPU_BUDGET_BYTES, GPU_BUDGET_MIN_MIB } from "../config.js";

const STORAGE_KEY = "teca.viewer.gpu-rendering-bytes";
const MIB = 1024 * 1024;

function validateGpuBudget(bytes) {
  const mib = bytes / MIB;
  if (!Number.isSafeInteger(mib) || mib < GPU_BUDGET_MIN_MIB || !Number.isSafeInteger(bytes * 8)) {
    throw new Error(`Live GPU budget must be at least ${GPU_BUDGET_MIN_MIB} whole MiB, not ${mib}.`);
  }
  return bytes;
}

export function readGpuBudget(storage) {
  const stored = storage.getItem(STORAGE_KEY);
  return stored === null ? GPU_BUDGET_BYTES : validateGpuBudget(Number(stored));
}

export function initGpuBudgetControls({ input, apply, status, bytes, applyBudget, storage }) {
  input.value = String(bytes / MIB);
  apply.addEventListener("click", () => {
    try {
      const next = validateGpuBudget(Number(input.value) * MIB);
      applyBudget(next);
      storage.setItem(STORAGE_KEY, String(next));
      status.textContent = `Live GPU budget: ${input.value} MiB. Views are refining within this shared limit.`;
    } catch (error) {
      status.textContent = error.message;
    }
  });
}
