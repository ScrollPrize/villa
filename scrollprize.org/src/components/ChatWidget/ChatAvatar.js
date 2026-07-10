import React from "react";

// Virtual Philodemus's mark: a scroll seen end-on — the CT cross-section
// spiral that is the project's most recognizable visual. Drawn as a single
// Archimedean spiral (2.6 turns), strokes in currentColor so each context
// tints it via CSS.
const SPIRAL =
  "M14.9 16.4 L14.6 16.0 L14.4 15.5 L14.4 14.9 L14.7 14.2 L15.3 13.7 L16.0 13.3 " +
  "L16.9 13.2 L17.9 13.4 L18.8 14.0 L19.5 14.9 L19.9 16.0 L19.9 17.3 L19.5 18.6 " +
  "L18.7 19.7 L17.5 20.6 L16.0 21.1 L14.3 21.1 L12.7 20.5 L11.3 19.4 L10.2 17.9 " +
  "L9.7 16.0 L9.7 14.0 L10.5 12.0 L11.8 10.3 L13.7 9.0 L16.0 8.4 L18.4 8.6 " +
  "L20.7 9.5 L22.7 11.1 L24.1 13.4 L24.8 16.0 L24.6 18.8 L23.5 21.5 L21.6 23.7 " +
  "L19.0 25.3 L16.0 26.0 L12.8 25.8 L9.8 24.5 L7.3 22.3 L5.5 19.4 L4.8 16.0 " +
  "L5.1 12.4 L6.5 9.1 L9.0 6.3 L12.2 4.4 L16.0 3.5 L19.9 3.9 L23.6 5.5 " +
  "L26.7 8.2 L28.8 11.8 L29.7 16.0 L29.3 20.3";

export default function ChatAvatar({ size = 36, className = "" }) {
  return (
    <span
      className={`vc-chat-avatar ${className}`.trim()}
      style={{ width: size, height: size }}
      aria-hidden="true"
    >
      <svg
        width={Math.round(size * 0.62)}
        height={Math.round(size * 0.62)}
        viewBox="0 0 33 33"
        fill="none"
      >
        <path
          d={SPIRAL}
          stroke="currentColor"
          strokeWidth="1.9"
          strokeLinecap="round"
          strokeLinejoin="round"
        />
      </svg>
    </span>
  );
}
