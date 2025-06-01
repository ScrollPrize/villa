import React from "react";

const AnimatedArrow = ({ text, button }) => (
  <div className={`flex ${button ? "" : "opacity-60"} text-sm`}>
    <div className="hidden sm:block uppercase font-bold tracking-wider mr-1 group-hover:mr-3 transition-all ease-in-out duration-300">
      {text}
    </div>
    <div className="block sm:hidden uppercase font-bold tracking-wider mr-1">
      {text}
    </div>
    <img
      src={
        button
          ? "/img/landing/arrow-right.svg"
          : "/img/landing/arrow-right-white.svg"
      }
    />
  </div>
);

const TopCard = ({ title, subtext, href, imageSrc, useArrow = false }) => {
  const cardContent = (
    <div
      className="h-auto md:h-28 relative rounded-2xl border border-[#FFFFFF20] bg-[#131114bf] group-hover:-translate-y-2 transition-transform ease-in-out duration-300 flex flex-col overflow-hidden"
      style={{
        boxShadow:
          "0px 2.767px 2.214px 0px rgba(0,0,0,0.09), 0px 6.65px 5.32px 0px rgba(0,0,0,0.13), 0px 12.522px 10.017px 0px rgba(0,0,0,0.16), 0px 22.336px 17.869px 0px rgba(0,0,0,0.19), 0px 41.778px 33.422px 0px rgba(0,0,0,0.23), 0px 100px 80px 0px rgba(0,0,0,0.32)",
      }}
    >
      <div className="flex flex-col py-3 md:py-2.5 px-4 md:px-5">
        <h3 className="text-base sm:text-lg md:text-xl text-white mt-0 mb-1 tracking-tighter leading-[90%] flex-grow">
          {title}
        </h3>
        {subtext && (
          <>
            <p className="text-xs md:hidden">{subtext}</p>
            {useArrow ? (
              <div className="hidden md:block">
                <AnimatedArrow text={subtext} />
              </div>
            ) : (
              <p className="hidden md:block text-sm">{subtext}</p>
            )}
          </>
        )}
      </div>
      {imageSrc && (
        <img
          className="absolute top-[50px] right-0 max-w-[190px] w-full h-auto object-contain"
          src={imageSrc}
          alt=""
        />
      )}
    </div>
  );

  return (
    <a
      className="cursor-pointer group hover:no-underline"
      href={href}
    >
      {cardContent}
    </a>
  );
};

export default TopCard;
