---
title: "Scrolls"
hide_table_of_contents: true
---

<head>
  <html data-theme="dark" />

  <meta
    name="description"
    content="A $1,000,000+ machine learning and computer vision competition"
  />

  <meta property="og:type" content="website" />
  <meta property="og:url" content="https://scrollprize.org" />
  <meta property="og:title" content="Vesuvius Challenge" />
  <meta
    property="og:description"
    content="A $1,000,000+ machine learning and computer vision competition"
  />
  <meta
    property="og:image"
    content="https://scrollprize.org/img/social/opengraph.jpg"
  />

  <meta property="twitter:card" content="summary_large_image" />
  <meta property="twitter:url" content="https://scrollprize.org" />
  <meta property="twitter:title" content="Vesuvius Challenge" />
  <meta
    property="twitter:description"
    content="A $1,000,000+ machine learning and computer vision competition"
  />
  <meta
    property="twitter:image"
    content="https://scrollprize.org/img/social/opengraph.jpg"
  />
</head>

3D X-ray micro-CT scans of the scrolls were performed with the [I12 Beamline](https://www.diamond.ac.uk/Instruments/Imaging-and-Microscopy/I12.html) at the Diamond Light Source synchrotron. Scanning capabilities include different combinations of resolutions (3.24µm, 7.91µm) and energy levels (53keV, 70keV, 88keV, 90keV, 105keV). The scan conditions for each scroll are listed below.

Scan outputs are "volumes", provided as a stack of 2D .tif files, or horizontal slices from bottom to top.
The file size and number of .tif files for each volume vary depending on the physical size of each scroll.

For more technical details, see [EduceLab-Scrolls: Verifiable Recovery of Text from Herculaneum Papyri using X-ray CT](https://arxiv.org/abs/2304.02084) and [EduceLab Herculaneum Scroll Data (2023) Info Sheet](https://drive.google.com/file/d/1I6JNrR6A9pMdANbn6uAuXbcDNwjk8qZ2/view?usp=sharing).

You can find the full scroll data on the [data server](https://dl.ash2txt.org/) in the [`full-scrolls/`](https://dl.ash2txt.org/full-scrolls/) folder.

At 3.24µm and 7.91µm resolution, the data files are big. We believe this resolution is necessary to detect ink, as suggested by the excellent paper [From invisibility to readability: Recovering the ink of Herculaneum](https://journals.plos.org/plosone/article/file?id=10.1371/journal.pone.0215775&type=printable).

## Scroll 1 (PHerc. Paris 4)

From the [Institut de France](https://www.institutdefrance.fr/en/home/).
The scroll for which we have by far the most [segments](data_segments), and in which the [first letters](firstletters) have been discovered and the [2023 Grand Prize](grandprize) was claimed.

<div className="flex w-[100%]">
  <div className="w-[100%] mb-2 mr-2"><img src="/img/overview/scroll1-actual-new.webp" className="w-[100%]"/><figcaption className="mt-0">Scroll 1 (PHerc. Paris. 4)</figcaption></div>
</div>
<iframe className="w-[100%] max-w-[500px] mb-4 aspect-square" src="https://www.youtube.com/embed/cY5BIxkf5m0"  title="YouTube video player" frameBorder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowFullScreen></iframe>

Volumes:

* [20230205180739](https://dl.ash2txt.org/full-scrolls/Scroll1/PHercParis4.volpkg/volumes/20230205180739/) (scroll top half): 7.91µm, 54keV, 14,376 x 128MB .tif files. Total size: 1.84 TB
* [20230206171837](https://dl.ash2txt.org/full-scrolls/Scroll1/PHercParis4.volpkg/volumes/20230206171837/) (scroll bottom half): 7.91µm, 54keV, 10,532 x 130MB .tif files. Total size: 1.37 TB

## Scroll 2 (PHerc. Paris 3)

From the [Institut de France](https://www.institutdefrance.fr/en/home/).
Has proven harder to segment due to many compressed layers.
The main volume has a scanning artifact in the middle of the volume, but much analysis is still possible.

<div className="flex w-[100%]">
  <div className="w-[100%] mb-2"><img src="/img/overview/scroll2-small-actual.webp" className="w-[100%]"/><figcaption className="mt-0">Scroll 2 (PHerc. Paris. 3)</figcaption></div>
</div>
<iframe className="w-[100%] max-w-[500px] mb-4 aspect-square" src="https://www.youtube.com/embed/RD-xyI8zefY"  title="YouTube video player" frameBorder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowFullScreen></iframe>

Volumes:

* [20230206082907](https://dl.ash2txt.org/full-scrolls/Scroll2/PHercParis3.volpkg/volumes/20230206082907/) (scroll top half): 7.91µm, 54keV, 6,586 x 191MB .tif files. Total size: 1.26 TB
* [20230210143520](https://dl.ash2txt.org/full-scrolls/Scroll2/PHercParis3.volpkg/volumes/20230210143520/) (scroll bottom half): 7.91µm, 54keV, 14,428 x 243MB .tif files. Total size: 3.50 TB
* [20230212125146](https://dl.ash2txt.org/full-scrolls/Scroll2/PHercParis3.volpkg/volumes/20230212125146/) (slab within top half): 7.91µm, 88keV, 1,610 x 189MB .tif files. Total size: 304 GB

## Scroll 3 (PHerc. 332)

From the [Biblioteca Nazionale di Napoli](https://www.bnnonline.it/it/121/officina-dei-papiri-ercolanesi).
A smaller scroll known as a *midollo* (marrow), left over from physical unwrapping attempts of the larger original scroll.
Due to the small size, could be scanned in entirety at extremely high resolution (3.24µm).

<div className="flex w-[100%]">
  <div className="sm:w-[35.5%] mb-2 mr-2"><img src="/img/overview/PHerc332.webp" className="w-[100%]"/><figcaption className="mt-[0]">Scroll 3 (PHerc. 332)</figcaption></div>
</div>
<iframe className="w-[100%] max-w-[500px] mb-4 aspect-square" src="https://www.youtube.com/embed/58mBW1hACuA"  title="YouTube video player" frameBorder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowFullScreen></iframe>

Volumes:

* [20231027191953](https://dl.ash2txt.org/full-scrolls/Scroll3/PHerc332.volpkg/volumes/20231027191953/) (entire scroll): 3.24µm, 53keV, 22,941 x 178MB .tif files. Total size: 4.07 TB
* [20231117143551](https://dl.ash2txt.org/full-scrolls/Scroll3/PHerc332.volpkg/volumes/20231117143551/) (entire scroll): 7.91µm, 53keV, 9,778 x 24MB .tif files. Total size: 236 GB
* [20231201141544](https://dl.ash2txt.org/full-scrolls/Scroll3/PHerc332.volpkg/volumes/20231201141544/) (entire scroll): 3.24µm, 70keV, 22,932 x 178MB .tif files. Total size: 4.06 TB

## Scroll 4 (PHerc. 1667)

From the [Biblioteca Nazionale di Napoli](https://www.bnnonline.it/it/121/officina-dei-papiri-ercolanesi).
Similar in size to Scroll 3.

<div className="flex w-[100%]">
  <div className="sm:w-[55%] mb-2"><img src="/img/overview/PHerc1667.webp" className="w-[100%]"/><figcaption className="mt-[0]">Scroll 4 (PHerc. 1667)</figcaption></div>
</div>
<iframe className="w-[100%] max-w-[500px] mb-4 aspect-square" src="https://www.youtube.com/embed/SyCZG6dc29c"  title="YouTube video player" frameBorder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowFullScreen></iframe>

Volumes:

* [20231107190228](https://dl.ash2txt.org/full-scrolls/Scroll4/PHerc1667.volpkg/volumes/20231107190228/) (entire scroll): 3.24µm, 88keV, 26,391 x 130MB .tif files. Total size: 3.41 TB
* [20231117161658](https://dl.ash2txt.org/full-scrolls/Scroll4/PHerc1667.volpkg/volumes/20231117161658/) (entire scroll): 7.91µm, 53keV, 11,174 x 23MB .tif files. Total size: 257 GB

## Scroll 5 (PHerc. 172)

From the [Bodleian Library](https://www.bodleian.ox.ac.uk/home) in Oxford.
Similar in size to Scroll 1.

<div className="flex w-[100%]">
  <div className="sm:w-[55%] mb-2"><img src="/img/overview/PHerc172.webp" className="w-[100%]"/><figcaption className="mt-[0]">Scroll 5 (PHerc. 172)</figcaption></div>
</div>

Volumes:

* [20241024131838](https://dl.ash2txt.org/full-scrolls/Scroll5/PHerc172.volpkg/volumes/20241024131838/) (entire scroll): 7.91µm, 53keV, 21,000 x 122MB .tif files. Total size: 2.56 TB
