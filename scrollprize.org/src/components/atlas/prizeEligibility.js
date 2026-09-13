import eligibility from "@site/src/data/prizeEligibility.json";

// Scroll-level view of src/data/prizeEligibility.json (the per-prize
// {scroll, volume} lists that EligibleVolumes renders on the prizes page):
// a scroll is "prize eligible" if any open prize lists at least one of its
// volumes. The data browser surfaces this as a plain scroll-level flag —
// which prize, and which volume, stays on the prizes page.

export const PRIZE_ELIGIBLE_SCROLLS = new Set(
  Object.values(eligibility).flatMap((entries) => entries.map((e) => e.scroll))
);

export function isPrizeEligible(scrollId) {
  return PRIZE_ELIGIBLE_SCROLLS.has(scrollId);
}
