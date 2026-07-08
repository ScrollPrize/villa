import React from "react";

// Pipeline thumbnail: quiet by default, framed media (hairline + 8px radius).
// The current step (highlight) reads as active nav: surface row + ember label.
// No shadows, no translate — hover is a surface tint only.
function Thumb({ href, highlight, label, videoSrc, poster, imgSrc }) {
  return (
    <a
      href={href}
      className={`mb-2 flex flex-col items-center w-[100px] sm:w-[150px] min-w-0 relative box-content p-2 sm:p-4 sm:pb-2 rounded-lg border border-solid hover:bg-surface hover:no-underline ${
        highlight ? "bg-surface border-line" : "border-transparent"
      }`}
    >
      {/* square media box */}
      <div className="relative w-full aspect-square rounded-lg overflow-hidden border border-solid border-line mb-2">
        {videoSrc ? (
          <video
            autoPlay
            playsInline
            loop
            muted
            poster={poster}
            className="absolute inset-0 w-full h-full object-cover"
          >
            <source src={videoSrc} type="video/webm" />
          </video>
        ) : (
          <img
            src={imgSrc}
            alt={label}
            loading="lazy"
            decoding="async"
            className="absolute inset-0 w-full h-full object-cover"
          />
        )}
      </div>
      <div
        className={`text-sm ${
          highlight ? "text-accent font-semibold" : "text-dim"
        }`}
      >
        {label}
      </div>
    </a>
  );
}

// `links` optionally re-targets each thumb (e.g. the Open Problems post points
// them at its own section anchors, turning the strip into a visual TOC).
// Defaults keep the tutorial destinations used on the tutorial pages.
export function TutorialsTop({ highlightId, links = {} } = {}) {
  return (
    // One row on ≥sm: thumbs shrink (min-w-0) instead of wrapping; the
    // arrows are fixed-width. Phones keep the centered wrap.
    <div className="mx-[-16px] sm:mx-0 flex flex-wrap sm:flex-nowrap items-start mb-4 text-center justify-center sm:justify-start">
      <Thumb
        href={links.scanning || "/tutorial1"}
        label="Scanning"
        videoSrc="/img/tutorial-thumbs/top-scanning-small.webm"
        poster="/img/tutorial-thumbs/top-scanning-small.webp"
        highlight={highlightId == 2}
      />

      <div className="hidden sm:flex mx-2 mb-2 flex-col items-center">
        <div className="relative leading-[150px] py-4 w-[16px] text-center text-faint">→</div>
        <div className="text-sm">&nbsp;</div>
      </div>

      {/* This one uses the JPG image and is square */}
      <Thumb
        href={links.representation || "/tutorial2"}
        label="Representation"
        imgSrc="/img/segmentation/normals_z0022.jpg"
        highlight={highlightId == 3}
      />

      <div className="hidden sm:flex mx-2 mb-2 flex-col items-center">
        <div className="relative leading-[150px] py-4 w-[16px] text-center text-faint">→</div>
        <div className="text-sm">&nbsp;</div>
      </div>

      <Thumb
        href={links.segmentation || "/segmentation"}
        label="Segmentation and Flattening"
        videoSrc="/img/tutorial-thumbs/top-segmentation-small.webm"
        poster="/img/tutorial-thumbs/top-segmentation-small.webp"
        highlight={highlightId == 4}
      />

      <div className="hidden sm:flex mx-2 mb-2 flex-col items-center">
        <div className="relative leading-[150px] py-4 w-[16px] text-center text-faint">→</div>
        <div className="text-sm">&nbsp;</div>
      </div>

      <Thumb
        href={links.ink || "/tutorial5"}
        label="Ink Detection"
        videoSrc="/img/tutorial-thumbs/top-prediction-small.webm"
        poster="/img/tutorial-thumbs/top-prediction-small.webp"
        highlight={highlightId == 5}
      />
    </div>
  );
}
