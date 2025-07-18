import React from 'react';

export function TutorialsTop({ highlightId } = {}) {
    return (
        <div className="mx-[-16px] sm:mx-0 flex flex-wrap items-start mb-4 text-center justify-center sm:justify-start">
            <a href="/tutorial1" className={`mb-2 flex flex-col items-center w-[100px] sm:w-[150px] relative box-content p-2 sm:p-4 sm:pb-2 hover:bg-[#fefefe26] rounded-xl ${ highlightId == 2 ? "bg-[#fffefc30] hover:bg-[#fefefe45]" : "" }`}>
                <video autoPlay playsInline loop muted className="w-[100%] rounded-xl mb-2"  poster="/img/tutorial-thumbs/top-scanning-small.webp">
                <source src="/img/tutorial-thumbs/top-scanning-small.webm" type="video/webm"/>
                </video>
                <div className="text-sm">Scanning</div>
            </a>
            <div className="hidden sm:flex mx-2 mb-2 flex-col items-center">
                <div className="relative leading-[150px] py-4 w-[16px] text-center">→</div>
                <div className="text-sm">&nbsp;</div>
            </div>
            <a href="/segmentation" className={`mb-2 flex flex-col items-center w-[100px] sm:w-[150px] relative box-content p-2 sm:p-4 sm:pb-2 hover:bg-[#fefefe26] rounded-xl ${ highlightId == 3 ? "bg-[#fffefc30] hover:bg-[#fefefe45]" : "" }`}>
                <video autoPlay playsInline loop muted className="w-[100%] rounded-xl mb-2" poster="/img/tutorial-thumbs/top-segmentation-small.webp">
                <source src="/img/tutorial-thumbs/top-segmentation-small.webm" type="video/webm"/>
                </video>
                <div className="text-sm">Segmentation and Flattening</div>
            </a>
            <div className="hidden sm:flex mx-2 mb-2 flex-col items-center">
                <div className="relative leading-[150px] py-4 w-[16px] text-center">→</div>
                <div className="text-sm">&nbsp;</div>
            </div>
            <a href="/tutorial4" className={`mb-2 flex flex-col items-center w-[100px] sm:w-[150px] relative box-content p-2 sm:p-4 sm:pb-2 hover:bg-[#fefefe26] rounded-xl ${ highlightId == 4 ? "bg-[#fffefc30] hover:bg-[#fefefe45]" : "" }`}>
            <img className="max-h-[300px] m-2" src="/img/tutorials/fisherman.webp"/>
            <div className="text-sm">Segmentation - a different approach</div>
            </a>
            <div className="hidden sm:flex mx-2 mb-2 flex-col items-center">
                <div className="relative leading-[150px] py-4 w-[16px] text-center">→</div>
                <div className="text-sm">&nbsp;</div>
            </div>
            <a href="/tutorial5" className={`mb-2 flex flex-col items-center w-[100px] sm:w-[150px] relative box-content p-2 sm:p-4 sm:pb-2 hover:bg-[#fefefe26] rounded-xl ${ highlightId == 5 ? "bg-[#fffefc30] hover:bg-[#fefefe45]" : "" }`}>
                <video autoPlay playsInline loop muted className="w-[100%] rounded-xl mb-2" poster="/img/tutorial-thumbs/top-prediction-small3.webp">
                <source src="/img/tutorial-thumbs/top-prediction-small.webm" type="video/webm"/>
                </video>
                <div className="text-sm">Ink Detection</div>
            </a>
        </div>
        
    );
}
