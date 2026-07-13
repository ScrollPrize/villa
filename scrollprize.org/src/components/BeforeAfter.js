import React, { useState, useRef, useEffect } from "react";
import { SPIRAL } from "./ChatWidget/ChatAvatar";

const BeforeAfter = ({
  beforeImage,
  afterImage,
  altBefore = "Before",
  altAfter = "After",
  beforeLabel,
  afterLabel,
  heightClass = "h-80",
}) => {
  const [sliderPosition, setSliderPosition] = useState(50);
  const containerRef = useRef(null);
  const [isDragging, setIsDragging] = useState(false);

  const handleDrag = (event) => {
    if (!isDragging) return;
    const container = containerRef.current;
    const rect = container.getBoundingClientRect();
    const x = (event.clientX || event.touches[0].clientX) - rect.left;
    const position = (x / rect.width) * 100;
    setSliderPosition(Math.min(Math.max(position, 0), 100));
  };

  const handleMouseDown = () => {
    setIsDragging(true);
  };

  const handleMouseUp = () => {
    setIsDragging(false);
  };

  useEffect(() => {
    document.addEventListener("mouseup", handleMouseUp);
    document.addEventListener("mousemove", handleDrag);
    document.addEventListener("touchend", handleMouseUp);
    document.addEventListener("touchmove", handleDrag);
    return () => {
      document.removeEventListener("mouseup", handleMouseUp);
      document.removeEventListener("mousemove", handleDrag);
      document.removeEventListener("touchend", handleMouseUp);
      document.removeEventListener("touchmove", handleDrag);
    };
  }, [isDragging]);

  return (
    <div
      ref={containerRef}
      className={`${heightClass} rounded-lg border border-solid border-line relative inline-block overflow-hidden cursor-col-resize w-full max-w-4xl select-none`}
      onMouseDown={handleMouseDown}
      onTouchStart={handleMouseDown}
      style={{
        WebkitUserSelect: "none",
      }}
    >
      <img
        src={afterImage}
        alt={altAfter}
        loading="lazy"
        decoding="async"
        className="absolute top-0 left-0 w-full h-full object-cover pointer-events-none"
      />

      <img
        src={beforeImage}
        alt={altBefore}
        loading="lazy"
        decoding="async"
        className="absolute top-0 left-0 w-full h-full object-cover pointer-events-none"
        style={{
          clipPath: `inset(0 ${100 - sliderPosition}% 0 0)`,
        }}
      />

      {beforeLabel && (
        <div className="absolute top-2 left-2 rounded px-2 py-0.5 text-xs text-dim bg-bg border border-solid border-line pointer-events-none">
          {beforeLabel}
        </div>
      )}
      {afterLabel && (
        <div className="absolute top-2 right-2 rounded px-2 py-0.5 text-xs text-dim bg-bg border border-solid border-line pointer-events-none">
          {afterLabel}
        </div>
      )}

      {/* 2px ember divider + 24px round handle with hairline ring (spec §3) */}
      <div
        className="absolute top-0 bottom-0 w-0.5 bg-accent"
        style={{ left: `${sliderPosition}%`, cursor: "col-resize" }}
      >
        <div className="absolute top-1/2 -translate-y-1/2 left-1/2 -translate-x-1/2 w-7 h-7 border border-solid border-line bg-bg rounded-full flex items-center justify-center">
          <svg width="19" height="19" viewBox="0 0 33 33" fill="none" className="text-accent">
            <path
              d={SPIRAL}
              stroke="currentColor"
              strokeWidth="3"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          </svg>
        </div>
      </div>
    </div>
  );
};

export default BeforeAfter;
