import React from "react";
import { usePluginData } from "@docusaurus/useGlobalData";

const usd = new Intl.NumberFormat("en-US", {
  style: "currency",
  currency: "USD",
  maximumFractionDigits: 0,
});

// Banner for the top of /prizes: the total open prize pool (computed from
// this very page's frontmatter via plugins/prizes-data.js) + the 2027
// deadline. Ember-bordered card, whole thing links to the Grand Prize.
export default function PrizePoolBanner() {
  const { prizes = [] } = usePluginData("prizes-data") || {};
  if (!prizes.length) return null;
  const total = prizes.reduce((sum, p) => sum + p.amount, 0);
  return (
    <a
      href="#2027-grand-prize"
      className="vc-card flex flex-wrap items-baseline gap-x-4 gap-y-1 my-6 hover:bg-raised hover:no-underline"
      style={{ borderColor: "var(--vc-accent)" }}
    >
      <span className="vc-label">Open prize pool</span>
      <span
        className="vc-nums text-accent font-bold"
        style={{ fontFamily: "var(--vc-mono)", fontSize: "1.75rem" }}
      >
        {usd.format(total)}
      </span>
      <span className="text-dim text-sm">
        Grand Prize deadline June 25th, 2027&nbsp;→
      </span>
    </a>
  );
}
