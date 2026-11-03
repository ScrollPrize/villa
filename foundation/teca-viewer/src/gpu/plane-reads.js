import { PREFETCH_CHUNK_PRIORITY, SHOWN_CHUNK_PRIORITY_LIMIT } from "../config.js";
import { chunkKey, visibleChunkPriority } from "./addresses.js";
import { presentationChunkRequest } from "./presentation-bricks.js";

function fetchStage(fallbackDepth, fetchOrder) {
  if (fetchOrder > 0) return 4;
  if (fallbackDepth > 0) return -2 * fallbackDepth + Number(fetchOrder < 0);
  return fetchOrder < 0 ? 0 : 1;
}

function fetchPriority(address, coarse, shown) {
  if (address.fetchOrder > 0) return PREFETCH_CHUNK_PRIORITY + visibleChunkPriority(address.priority);
  if (!shown) return SHOWN_CHUNK_PRIORITY_LIMIT - 1;
  return coarse ? 0 : visibleChunkPriority(address.priority);
}

export function retainPlaneReads(volume, channel, planePlans) {
  const requestsByScale = new Map();
  for (const plan of planePlans) {
    for (const level of plan.levels) {
      let requests = requestsByScale.get(level.scale);
      if (requests === undefined) {
        requests = new Map();
        requestsByScale.set(level.scale, requests);
      }
      for (const address of level.fetchAddresses) {
        const request = presentationChunkRequest(level.scale, address.chunk_ZYX);
        const key = request.chunk_ZYX.join("/");
        const priority = fetchPriority(
          address,
          plan.levels.length === 1 || level.scaleIndex > plan.scaleIndex,
          plan.shown,
        );
        const previous = requests.get(key);
        if (previous === undefined || priority < previous.priority) {
          requests.set(key, { chunk_ZYX: request.chunk_ZYX, priority });
        }
      }
    }
  }
  for (const [scale, requests] of requestsByScale)
    volume.retainChunkRequests([...requests.values()], scale.sourceScale ?? scale, channel);
}

export function planeReads(planePlans) {
  const shownFirst = planePlans.toSorted((left, right) => Number(right.shown) - Number(left.shown));
  const tasksByPlane = new Map(shownFirst.map(({ plane }) => [plane, []]));
  const tasksByKey = new Map();
  for (const plan of shownFirst) {
    for (const level of plan.levels) {
      const fallbackDepth = level.scaleIndex - plan.scaleIndex;
      const coarse = plan.levels.length === 1 || fallbackDepth > 0;
      for (const address of level.fetchAddresses) {
        const residentKey = chunkKey(level.scale, address.chunk_ZYX);
        if (!level.atlas.inWorkingSet(residentKey) || level.atlas.get(residentKey) !== null) continue;
        const key = `${level.scaleKey}\0${residentKey}`;
        const fetchOrder = address.fetchOrder;
        const stage = fetchStage(fallbackDepth, fetchOrder);
        const priority = fetchPriority(address, coarse, plan.shown);
        const previous = tasksByKey.get(key);
        if (previous !== undefined) {
          if (stage < previous.stage) {
            previous.fetchOrder = fetchOrder;
            previous.shown = plan.shown;
          } else if (stage === previous.stage) {
            previous.fetchOrder = Math.min(previous.fetchOrder, fetchOrder);
            previous.shown ||= plan.shown;
          }
          previous.stage = Math.min(previous.stage, stage);
          previous.priority = Math.min(previous.priority, priority);
          previous.coarse ||= coarse;
          continue;
        }
        const task = {
          key,
          residentKey,
          scaleKey: level.scaleKey,
          scale: level.scale,
          chunk_ZYX: address.chunk_ZYX,
          priority,
          fetchOrder,
          stage,
          coarse,
          shown: plan.shown,
        };
        tasksByKey.set(key, task);
        tasksByPlane.get(plan.plane).push(task);
      }
    }
  }
  let maximumTasks = 0;
  for (const tasks of tasksByPlane.values()) {
    tasks.sort(
      (left, right) =>
        left.stage - right.stage ||
        left.fetchOrder - right.fetchOrder ||
        Number(right.coarse) - Number(left.coarse) ||
        left.priority - right.priority,
    );
    maximumTasks = Math.max(maximumTasks, tasks.length);
  }
  const tasks = [];
  for (let index = 0; index < maximumTasks; index++) {
    for (const paneTasks of tasksByPlane.values()) {
      if (index < paneTasks.length) tasks.push(paneTasks[index]);
    }
  }
  return tasks.sort(
    (left, right) =>
      Number(left.fetchOrder > 0) - Number(right.fetchOrder > 0) ||
      Number(right.shown) - Number(left.shown) ||
      left.stage - right.stage ||
      left.fetchOrder - right.fetchOrder,
  );
}
