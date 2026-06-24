import React from "react";
import ScrollCard from "./ScrollCard";

// The responsive card grid. The reference uses id="grid"; per the styling
// contract this native rebuild uses className="atlas-grid" instead. Samples are
// sorted most-progress-first (defensively, in case the feed isn't presorted).

export default function ScrollGrid({ scrolls }) {
  const items = (scrolls || [])
    .slice()
    .sort(
      (a, b) =>
        ((b.progress || {}).score || 0) - ((a.progress || {}).score || 0)
    );

  return (
    <div className="atlas-grid">
      {items.map((s) => (
        <ScrollCard key={s.id} scroll={s} />
      ))}
    </div>
  );
}
