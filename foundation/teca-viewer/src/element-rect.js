const rectangles = new Map();

export function elementRect(element) {
  let rect = rectangles.get(element);
  if (rect === undefined) {
    if (rectangles.size === 0) queueMicrotask(() => rectangles.clear());
    rect = element.getBoundingClientRect();
    rectangles.set(element, rect);
  }
  return rect;
}

export function invalidateElementRects() {
  rectangles.clear();
}
