import React from 'react';

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
  
  const baseClasses = `${maxWidth} mr-4 mb-4 text-gray-100 bg-[#444] hover:bg-[#555] hover:text-[unset] p-4 rounded-lg flex flex-col justify-between ${className}`;
  
  const renderMedia = () => {
    if (mediaType === 'video') {
      return (
        <video 
          autoPlay 
          playsInline 
          muted 
          loop 
          className={`w-[100%] ${finalImageClassName}`}
        >
          <source src={mediaSrc} type={videoType} />
        </video>
      );
    } else {
      return (
        <img 
          src={mediaSrc} 
          alt={mediaAlt}
          className={finalImageClassName}
        />
      );
    }
  };

  return (
    <a className={baseClasses} href={href}>
      <div className="mb-4">
        <div className="text-sm font-bold text-gray-300">{prizeAmount}</div>
        <span className="font-bold">{title}:</span>
        {description && (
          <> {description}</>
        )}
      </div>
      {mediaSrc && renderMedia()}
    </a>
  );
};

export default PrizeCard; 