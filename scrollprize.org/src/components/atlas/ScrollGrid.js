import React from "react";
import ScrollCard from "./ScrollCard";

// The responsive card grid. The reference uses id="grid"; per the styling
// contract this native rebuild uses className="atlas-grid". Cards render in the
// order given by AtlasBrowser (which owns the search/sort/filter controls) and
// are keyed by `id` so reordering never churns the shared WebGL canvas registry.

export default function ScrollGrid({ scrolls }) {
  const items = scrolls || [];
  return (
    <div className="atlas-grid">
      {items.map((s) => (
        <ScrollCard key={s.id} scroll={s} />
      ))}
    </div>
  );
}
