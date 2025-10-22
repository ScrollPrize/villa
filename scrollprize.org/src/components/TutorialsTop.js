import React from "react";

function Thumb({ href, highlight, label, videoSrc, poster, imgSrc }) {
  return (
    <a
      href={href}
      className={`mb-2 flex flex-col items-center w-[100px] sm:w-[150px] relative box-content p-2 sm:p-4 sm:pb-2 hover:bg-[#fefefe26] rounded-xl ${
        highlight ? "bg-[#fffefc30] hover:bg-[#fefefe45]" : ""
      }`}
    >
      {/* square media box */}
      <div className="relative w-full aspect-square rounded-xl overflow-hidden mb-2">
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
      <div className="text-sm">{label}</div>
    </a>
  );
}

export function TutorialsTop({ highlightId } = {}) {
  return (
    <div className="mx-[-16px] sm:mx-0 flex flex-wrap items-start mb-4 text-center justify-center sm:justify-start">
      <Thumb
        href="/tutorial1"
        label="Scanning"
        videoSrc="/img/tutorial-thumbs/top-scanning-small.webm"
        poster="/img/tutorial-thumbs/top-scanning-small.webp"
        highlight={highlightId == 2}
      />

      <div className="hidden sm:flex mx-2 mb-2 flex-col items-center">
        <div className="relative leading-[150px] py-4 w-[16px] text-center">→</div>
        <div className="text-sm">&nbsp;</div>
      </div>

      {/* This one uses the JPG image and is square */}
      <Thumb
        href="/tutorial2"
        label="Representation"
        imgSrc="/img/segmentation/normals_z0022.jpg"
        highlight={highlightId == 3}
      />

      <div className="hidden sm:flex mx-2 mb-2 flex-col items-center">
        <div className="relative leading-[150px] py-4 w-[16px] text-center">→</div>
        <div className="text-sm">&nbsp;</div>
      </div>

      <Thumb
        href="/segmentation"
        label="Segmentation and Flattening"
        videoSrc="/img/tutorial-thumbs/top-segmentation-small.webm"
        poster="/img/tutorial-thumbs/top-segmentation-small.webp"
        highlight={highlightId == 4}
      />

      <div className="hidden sm:flex mx-2 mb-2 flex-col items-center">
        <div className="relative leading-[150px] py-4 w-[16px] text-center">→</div>
        <div className="text-sm">&nbsp;</div>
      </div>

      <Thumb
        href="/tutorial5"
        label="Ink Detection"
        videoSrc="/img/tutorial-thumbs/top-prediction-small.webm"
        poster="/img/tutorial-thumbs/top-prediction-small3.webp"
        highlight={highlightId == 5}
      />
    </div>
  );
}
