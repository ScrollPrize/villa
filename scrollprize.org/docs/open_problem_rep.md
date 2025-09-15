<head>
  <html data-theme="dark" />

  <meta
    name="description"
    content="A $1,500,000+ machine learning and computer vision competition"
  />

  <meta property="og:type" content="website" />
  <meta property="og:url" content="https://scrollprize.org" />
  <meta property="og:title" content="Vesuvius Challenge" />
  <meta
    property="og:description"
    content="A $1,500,000+ machine learning and computer vision competition"
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
    content="A $1,500,000+ machine learning and computer vision competition"
  />
  <meta
    property="twitter:image"
    content="https://scrollprize.org/img/social/opengraph.jpg"
  />
</head>


### Representation
___
Representation is best thought of in the context of this Challenge as an idealized version of some scroll feature. A "perfect" representation of a scroll might be a well space archimedan spiral  

To create a readable 2d flattened papyrus sheet from a scroll volume, we have to first map our 3d surface to 2d space. To do this, we need to know _which points_ to map to 2d space, and in _what order_. The process is known as parameterization. Our goal in the representation step is to simplify these two questions.  

<div className="mb-4">
  <a target="_blank" href="/img/segmentation/arap-parameterization.png"><img src="/img/segmentation/arap-parameterization.png" className="w-[100%]"/></a>
  <figcaption className="mt-[-6px]">Results from ARAP on a cylindrical and sinusoidal surface [1].</figcaption>
</div>
[img](../static/img/segmentation/arap-parameterization.png)