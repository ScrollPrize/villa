import React from "react";

// The per-card progress block (".prog"). Ported from the reference renderer
// (index.html ~229-238): a row of 5 step segments coloured per the stages a
// sample has reached, plus a label showing the furthest stage and an optional
// "N% unrolled" readout.

const STAGE_KEYS = ["scanned", "segmented", "unrolled", "ink", "text"];
const STAGE_LABEL = {
  scanned: "Scanned",
  segmented: "Segmented",
  unrolled: "Unrolled",
  ink: "Ink detected",
  text: "Text recovered",
};

export default function PipelineStepper({ stages }) {
  const st = stages || { scanned: true };

  // Furthest reached stage label (defaults to "Scanned").
  const reached = STAGE_KEYS.filter((k) => st[k]);
  const furLabel = STAGE_LABEL[reached[reached.length - 1]] || "Scanned";
  const up = st.unrolledPct;
  const pctTxt =
    typeof up === "number" && up
      ? `${up}% unrolled`
      : typeof up === "string" && up
      ? `unrolled: ${up}`
      : "";

  return (
    <div className="prog">
      <div
        className="steps"
        role="img"
        aria-label={`pipeline: reached ${furLabel}`}
      >
        {STAGE_KEYS.map((k, i) => (
          <span
            key={k}
            className={st[k] ? `f${i + 1}` : ""}
            title={`${STAGE_LABEL[k]}${st[k] ? " ✓" : ""}`}
          />
        ))}
      </div>
      <div className="plabel">
        <b>{furLabel}</b>
        <span>{pctTxt}</span>
      </div>
    </div>
  );
}
