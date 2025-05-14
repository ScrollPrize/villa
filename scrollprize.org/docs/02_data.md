---
title: "The Data"
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

**To download:** Fill out the [registration form](https://forms.gle/HV1J6dJbmCB2z5QL8) and then visit the [data server](https://dl.ash2txt.org).

To learn more about the data, see the linked pages below.
Also be sure to check out:

* [EduceLab-Scrolls (2019)](https://arxiv.org/abs/2304.02084): technical paper describing the original data.
* [EduceLab Data Sheet (2023)](https://drive.google.com/file/d/1I6JNrR6A9pMdANbn6uAuXbcDNwjk8qZ2/view?usp=sharing): technical paper describing more recent scans added to the dataset.
* [Tutorials](tutorial): what to do with the data.
* Our libraries to access data in 1-2 lines of code: in [Python](https://github.com/ScrollPrize/villa/tree/main/vesuvius) (with [intro notebook](https://colab.research.google.com/github/ScrollPrize/villa/blob/main/vesuvius/notebooks/example1_data_access.ipynb)) and [in C](https://github.com/ScrollPrize/villa/tree/main/vesuvius-c)!

## Scrolls

Micro-CT scans of intact Herculaneum scrolls.
The mission is to virtually unwrap the contents of the scrolls from the CT scans, revealing the text hidden within.
Scroll 1 was used to win the 2023 Grand Prize, but 95% of the scroll remains unread!

[More information](data_scrolls)

<div>
  <table>
    <tbody>
      <tr>
        <td>
          <div className="flex flex-wrap max-w-[500px]">
            <div className="sm:w-[45%] mb-2 mr-2"><img src="/img/overview/scroll1-actual-new.webp" className="w-[100%]"/><figcaption className="mt-[-6px]">Scroll 1 (PHerc. Paris. 4)</figcaption></div>
            <div className="sm:w-[45%] mb-2"><img src="/img/overview/scroll2-small-actual.webp" className="w-[100%]"/><figcaption className="mt-[-6px]">Scroll 2 (PHerc. Paris. 3)</figcaption></div>
            <div className="sm:w-[35.5%] mb-2 mr-2"><img src="/img/overview/PHerc332.webp" className="w-[100%]"/><figcaption className="mt-[-6px]">Scroll 3 (PHerc. 332)</figcaption></div>
            <div className="sm:w-[55%] mb-2"><img src="/img/overview/PHerc1667.webp" className="w-[100%]"/><figcaption className="mt-[-6px]">Scroll 4 (PHerc. 1667)</figcaption></div>
            <div className="sm:w-[55%] mb-2"><img src="/img/overview/PHerc172.webp" className="w-[100%]"/><figcaption className="mt-[-6px]">Scroll 5 (PHerc. 172)</figcaption></div>
          </div>
        </td>
      </tr>
    </tbody>
  </table>
</div>

## Fragments

Micro-CT scans of detached scroll fragments.
Since the fragments have exposed text on their surfaces, they can be used as ground truth for machine learning-based ink detection approaches (see [Tutorial 5: Ink Detection](tutorial5)).

[More information](data_fragments)

<div>
  <table>
    <tbody>
      <tr>
        <td>
          <div className="flex flex-wrap">
            <div className="sm:w-[45%] max-w-[200px] mb-2 mr-2"><img src="/img/data/fr1.webp" className="w-[100%]"/><figcaption className="mt-[-6px]">Fragment 1 (PHerc. Paris. 2 Fr 47)</figcaption></div>
            <div className="sm:w-[45%] max-w-[200px] mb-2 mr-2"><img src="/img/data/fr2.webp" className="w-[100%]"/><figcaption className="mt-[-6px]">Fragment 2 (PHerc. Paris. 2 Fr 143)</figcaption></div>
            <div className="sm:w-[45%] max-w-[200px] mb-2 mr-2"><img src="/img/data/fr3.webp" className="w-[100%]"/><figcaption className="mt-[-6px]">Fragment 3 (PHerc. Paris. 1 Fr 34)</figcaption></div>
            <div className="sm:w-[45%] max-w-[200px] mb-2 mr-2"><img src="/img/data/fr4.webp" className="w-[100%]"/><figcaption className="mt-[-6px]">Fragment 4 (PHerc. Paris. 1 Fr 39)</figcaption></div>
            <div className="sm:w-[45%] max-w-[130px] mb-2 mr-2"><img src="/img/data/fr5.webp" className="w-[100%]"/><figcaption className="mt-[-6px]">Fragment 5 (PHerc. 1667 Cr 1 Fr 3)</figcaption></div>
            <div className="sm:w-[45%] max-w-[220px] mb-2 mr-2"><img src="/img/data/fr6.webp" className="w-[100%]"/><figcaption className="mt-[-6px]">Fragment 6 (PHerc. 51 Cr 4 Fr 48)</figcaption></div>
          </div>
        </td>
      </tr>
    </tbody>
  </table>
</div>

## Segments

Segmentation is the mapping of sheets of papyrus in a 3D X-ray volume. The resulting surface volumes can be used directly to look for ink.

[More information](data_segments)

<div>
  <table>
    <tbody>
      <tr>
        <td>
          <div className="flex flex-wrap max-w-[500px]">
            <div className="w-[100%] max-w-[400px] mb-4"><div className="overflow-hidden mb-2"><img loading="eager" src="/img/data/segmentation-animation.webp" className="w-[100%] mt-[-30px] mb-[-50px]"/></div><figcaption className="mt-[-6px]">Some segments from Scroll 1.</figcaption></div>
          </div>
        </td>
      </tr>
    </tbody>
  </table>
</div>
