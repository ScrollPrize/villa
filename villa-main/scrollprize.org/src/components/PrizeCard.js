import React from 'react';

// Flat prize card on the ONE card recipe (.vc-card: surface + hairline + 8px
// radius, hover = raised bg + stronger border — no shadows). One red maximum:
// the $ amount carries the accent; title/description stay neutral.
// Props are stable — all existing consumers (docs/15_winners.md etc.) keep working.
const PrizeCard = ({
  href,
  prizeAmount,
  title,
  description,
  mediaSrc,
  mediaType = 'image',
  mediaAlt = '',
  videoType = 'video/webm',
  wide = false,
  imageClassName = '',
  className = ''
}) => {
  const maxWidth = wide ? 'max-w-[632px]' : 'max-w-[200px]';
  const defaultImageClassName = wide ? 'max-w-[100%]' : '';
  const finalImageClassName = imageClassName || defaultImageClassName;

  const baseClasses = `vc-card ${maxWidth} mr-4 mb-4 flex flex-col justify-between hover:no-underline ${className}`;

  const renderMedia = () => {
    if (mediaType === 'video') {
      return (
        <video
          autoPlay
          playsInline
          muted
          loop
          className={`vc-media w-full ${finalImageClassName}`}
        >
          <source src={mediaSrc} type={videoType} />
        </video>
      );
    } else {
      return (
        <img
          src={mediaSrc}
          alt={mediaAlt}
          className={`vc-media ${finalImageClassName}`}
        />
      );
    }
  };

  return (
    <a className={baseClasses} href={href}>
      <div className="mb-4">
        <div className="text-sm font-semibold text-accent vc-nums">{prizeAmount}</div>
        <span className="font-semibold">{title}:</span>
        {description && (
          <> {description}</>
        )}
      </div>
      {mediaSrc && renderMedia()}
    </a>
  );
};

export default PrizeCard;
