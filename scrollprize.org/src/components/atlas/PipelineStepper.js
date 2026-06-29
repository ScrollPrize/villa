import React from "react";

// The per-card progress block (".prog"): a 4-stage pipeline —
// Scanned → Segmented → Ink detected → Text recovered. Segmentation and
// unrolling are merged into one "Segmented" stage (a surface is segmented
// before it can be unrolled, and unrolling is a subset of segmentation);
// per-scroll unrolling progress is kept separately as the "N% unrolled" readout.

const DISPLAY = [
  { key: "scanned", label: "Scanned", cls: "f1" },
  { key: "segmented", label: "Segmented", cls: "f2" },
  { key: "ink", label: "Ink detected", cls: "f4" },
  { key: "text", label: "Text recovered", cls: "f5" },
];

// "Segmented" is reached once a sample has any traced surface (segmented or
// unrolled); the rest map straight to their stage flag.
const reached = (st, key) =>
  key === "segmented" ? !!(st.segmented || st.unrolled) : !!st[key];

export default function PipelineStepper({ stages }) {
  const st = stages || { scanned: true };

  const done = DISPLAY.filter((d) => reached(st, d.key));
  const furLabel = (done[done.length - 1] || DISPLAY[0]).label;
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
        {DISPLAY.map((d) => {
          const on = reached(st, d.key);
          return (
            <span
              key={d.key}
              className={on ? d.cls : ""}
              title={`${d.label}${on ? " ✓" : ""}`}
            />
          );
        })}
      </div>
      <div className="plabel">
        <b>{furLabel}</b>
        <span>{pctTxt}</span>
      </div>
    </div>
  );
}
