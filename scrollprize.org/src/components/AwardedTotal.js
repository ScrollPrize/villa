import React from "react";
import { usePluginData } from "@docusaurus/useGlobalData";

// Total dollars awarded, computed at build time from the winners page's money
// headings (plugins/winners-data.js). Renders "$1,800,500", or "$1.8M+" with
// `compact`. Use anywhere in MDX/JSX instead of hardcoding the figure.
export default function AwardedTotal({ compact = false }) {
  const { awardedTotal } = usePluginData("winners-data") || {};
  if (!awardedTotal) return null;
  if (compact) {
    return <>${(Math.floor(awardedTotal / 100000) / 10).toFixed(1)}M+</>;
  }
  return (
    <>
      $
      {new Intl.NumberFormat("en-US", { maximumFractionDigits: 0 }).format(
        awardedTotal,
      )}
    </>
  );
}
