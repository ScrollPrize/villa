import React from "react";

// Flat "Obsidian" teaser card: the ONE card recipe (surface + hairline + 8px
// radius via .vc-card), no shadows, no translate on hover. Props are stable
// (title, subtext, href, imageSrc, useArrow) so existing consumers keep working.
const TopCard = ({ title, subtext, href, imageSrc, useArrow = false }) => (
  <a
    className="vc-card vc-card--flush relative flex h-auto md:h-28 flex-col overflow-hidden cursor-pointer hover:no-underline"
    href={href}
  >
    <div className="flex flex-col px-4 py-3">
      <h3 className="text-base md:text-lg font-semibold leading-snug mt-0 mb-1 flex-grow">
        {title}
      </h3>
      {subtext &&
        (useArrow ? (
          <span className="vc-cta">{subtext}</span>
        ) : (
          <p className="text-xs md:text-sm text-faint m-0">{subtext}</p>
        ))}
    </div>
    {imageSrc && (
      <img
        className="absolute top-[50px] right-0 max-w-[190px] w-full h-auto object-contain"
        src={imageSrc}
        alt=""
      />
    )}
  </a>
);

export default TopCard;
