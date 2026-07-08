---
title: "An entire Herculaneum scroll has been read for the first time"
sidebar_label: "First scroll read (Jun 2026)"
hide_table_of_contents: true
hide_title: true
---

<head>
  <html data-theme="dark" />

  <meta name="description" content="PHerc. 1667, sealed since the eruption of Vesuvius in 79 AD, has been virtually unwrapped and read from beginning to end — the first Herculaneum scroll recovered in full, without ever being opened." />

  <meta property="og:type" content="article" />
  <meta property="og:url" content="https://scrollprize.org/firstscroll" />
  <meta property="og:title" content="An entire Herculaneum scroll has been read for the first time" />
  <meta property="og:description" content="PHerc. 1667, sealed since the eruption of Vesuvius in 79 AD, has been virtually unwrapped and read from beginning to end — without ever being opened." />
  <meta property="og:image" content="https://scrollprize.org/img/firstscroll/og.webp" />

  <meta property="twitter:card" content="summary_large_image" />
  <meta property="twitter:url" content="https://scrollprize.org/firstscroll" />
  <meta property="twitter:title" content="An entire Herculaneum scroll has been read for the first time" />
  <meta property="twitter:description" content="PHerc. 1667 has been virtually unwrapped and read from beginning to end — without ever being opened." />
  <meta property="twitter:image" content="https://scrollprize.org/img/firstscroll/og.webp" />
</head>

import JsonLd from '@site/src/components/JsonLd';

<JsonLd data={{"@context":"https://schema.org","@type":"NewsArticle","headline":"An entire Herculaneum scroll has been read for the first time","datePublished":"2026-06-25","dateModified":"2026-06-25","author":{"@type":"Organization","name":"Vesuvius Challenge"},"publisher":{"@type":"Organization","name":"Vesuvius Challenge","logo":{"@type":"ImageObject","url":"https://scrollprize.org/img/social/opengraph.jpg"}},"image":"https://scrollprize.org/img/firstscroll/og.webp","mainEntityOfPage":"https://scrollprize.org/firstscroll"}} />

<figure className="!mt-0 !mb-6">
  <video autoPlay playsInline loop muted className="w-[100%] rounded-lg" poster="/img/firstscroll/hero-poster.webp">
    <source src="/img/firstscroll/hero-reveal.webm" type="video/webm" />
    <source src="/img/firstscroll/hero-reveal.mp4" type="video/mp4" />
  </video>
</figure>

<h1 className="color-white text-4xl md:text-7xl font-black !mb-2 leading-none tracking-tighter">We read an <span style={{
  background:
    "radial-gradient(53.44% 245.78% at 13.64% 46.56%, #F5653F 0%, #D53A17 100%)",
  WebkitBackgroundClip: "text",
  WebkitTextFillColor: "transparent",
  backgroundClip: "text",
  textFillColor: "transparent",
}}>entire scroll</span> — without ever opening it</h1>

<div className="md:text-3xl text-lg font-medium mt-6 mb-2 opacity-80 leading-none tracking-tight">PHerc. 1667, sealed since the eruption of Vesuvius in 79 AD, has been virtually unwrapped and read from beginning to end.</div>

<div className="opacity-60 mb-8 italic">June 25th, 2026</div>

**Read the preprint:** <a href="/pdf/main.pdf" target="_blank" rel="noopener"><em>Complete virtual unwrapping and reading of a rolled Herculaneum papyrus</em> (PDF)</a> and on <a href="https://arxiv.org/abs/2606.29085" target="_blank" rel="noopener">arXiv</a>. The data is openly available at [scrollprize.org/data](/data), and the code on [GitHub](https://github.com/ScrollPrize/villa).

For almost 2,000 years, the carbonized library of Herculaneum has kept a cruel bargain: its scrolls survived the eruption of Mount Vesuvius, but only by becoming too fragile to open. To read one was to destroy it. Hundreds of rolls have therefore remained sealed, their contents preserved yet unreachable.

Today that changes. We have **completely virtually unwrapped and read PHerc. 1667** — the scroll the Vesuvius Challenge community knows as Scroll 4 — without ever touching its pages. It is the first Herculaneum papyrus to be digitally unrolled and read in full, end to end, and made available for sustained scholarly study.

<figure className="my-6">
  <div className="relative rounded-lg overflow-hidden border border-solid border-[#FFFFFF20]">
    <div className="overflow-x-auto">
      <img src="/img/firstscroll/banner-strip.webp" data-zoom-src="/img/firstscroll/banner-full.webp" className="block !h-[200px] md:!h-[300px] w-auto max-w-none" alt="The complete unwrapped writing surface of PHerc. 1667, showing columns of ancient Greek text." />
    </div>
    {/* Scalebar overlay — pinned, stays in view while panning. Bar width = 10 cm at the displayed scale (depends on the h-[200px]/h-[300px] image heights above). */}
    <div className="pointer-events-none absolute left-3 bottom-3 flex flex-col items-start gap-1">
      <span className="text-white text-xs md:text-sm font-semibold leading-none px-1.5 py-1 rounded bg-[#000000a6]">10 cm</span>
      <div className="h-[5px] md:h-[6px] w-[179px] md:w-[269px] bg-white rounded-[2px] ring-1 ring-[#00000066]"></div>
    </div>
    {/* Credit overlay — pinned bottom-right */}
    <div className="pointer-events-none absolute right-3 bottom-3 text-[#ffffffe6] text-xs md:text-sm font-medium leading-none px-1.5 py-1 rounded bg-[#000000a6]">© Vesuvius Challenge 2026</div>
  </div>
  <figcaption className="mt-0">The complete writing surface of PHerc. 1667, virtually unwrapped — roughly 1.4 metres of papyrus and around twenty-two columns of Greek. Scroll sideways to pan; click to zoom. <a href="/img/firstscroll/banner-full.webp" target="_blank" rel="noopener">Download the high-resolution image.</a></figcaption>
</figure>

## From a sealed lump to a readable book

PHerc. 1667 began as a blackened, rolled mass of carbonized papyrus. To read it, we never unrolled it physically. Instead, we scanned it with high-resolution X-rays, reconstructed the wound sheet inside the volume, flattened it into a readable surface, and used machine learning to bring out the faint traces of ancient ink.

<figure className="">
  <img src="/img/firstscroll/fig1.webp" alt="PHerc. 1667 from sealed roll to readable text: a photograph of the carbonized roll, transverse and longitudinal CT cross-sections, and the unwrapped surface showing columns of Greek." />
  <figcaption className="mt-0">From object to text. The sealed, carbonized roll (top left); cross-sections through the X-ray scan revealing the spiraled sheet inside (top); and the unwrapped surface, where columns of Greek writing emerge as the ink signal is recovered (bottom).</figcaption>
</figure>

## Three sealed scrolls, three milestones

The work reaches beyond a single scroll. Alongside the complete reading of PHerc. 1667, the research establishes a method that holds up under independent checks and scales to other rolls.

### PHerc. 1667 — read in full

PHerc. 1667 is what survives of a larger roll: earlier attempts to open it by hand — in the nineteenth century, and again in 1969 and the 1980s — destroyed its outer layers and left only the compact inner core, about 8 cm of an original height of 19–24 cm. From that surviving portion we have now recovered and read the text **in full** — the lower parts of some twenty-two columns, transcribed and reviewed by papyrologists. It is the first time the preserved text of a rolled Herculaneum scroll has been read continuously, end to end, rather than in isolated words or patches.

The recovered text is a **philosophical treatise on ethics**, and the evidence points to a **Stoic work**: it turns on human nature, impulse, and the moral progress of human beings, and its final preserved column names **Aristocreon** — nephew and disciple of the great Stoic **Chrysippus** — which, together with the language and themes of the text, places it in a Stoic context and dates it to the 2nd century BC.

Because the papyrus is damaged, the readings are fragmentary, with gaps where the surface is lost. Even so, several passages can be read clearly for the first time in two thousand years:

<div className="border-l-4 border-solid border-0 pl-4 mb-4 italic border-l-[var(--ifm-color-primary)]">“…we will inquire into something, but we will not grasp it, if in some way we depart from ourselves and from our own nature…”</div>

<div className="border-l-4 border-solid border-0 pl-4 mb-4 italic border-l-[var(--ifm-color-primary)]">“Having…strained ourselves to the utmost through research and learning…possessing the same practical wisdom…”</div>

<div className="border-l-4 border-solid border-0 pl-4 mb-4 italic border-l-[var(--ifm-color-primary)]">“…such being the goods for us, even from the opposite evils there will be neither anything good — let alone beautiful — nor anything bad — let alone ugly — nor happiness…”</div>

<div className="opacity-60 italic mb-4 text-sm">Translated from the Greek; the full column-by-column transcription is in the <a href="/pdf/main.pdf" target="_blank" rel="noopener">preprint</a>.</div>

### PHerc. Paris 4 — ink made visible by higher resolution

In a second scroll — **PHerc. Paris 4, the scroll the Vesuvius Challenge community knows as Scroll 1** — a higher-resolution imaging technique makes the ink **directly visible inside the scroll itself**, in the three-dimensional X-ray data, for the first time. Segmented in 3D and projected back onto the unwrapped page, that ink matches the text read in the 2023 Grand Prize **one-to-one** — an independent confirmation, from better data, that the reading is real.

<figure className="">
  <img src="/img/firstscroll/fig2.webp" alt="Higher-resolution cross-section of PHerc. Paris 4 showing ink directly visible on the papyrus surface, with the ink segmented in three dimensions." />
  <figcaption className="mt-0">Ink made visible by higher resolution in PHerc. Paris 4 (Scroll 1). In a cross-section of the X-ray scan, the ink sits directly on the papyrus surface (left); it can then be segmented in three dimensions (red) and projected onto the unwrapped page.</figcaption>
</figure>

<figure className="">
  <img src="/img/firstscroll/fig3.webp" alt="PHerc. Paris 4: the 2023 Grand Prize reading compared with the 2024 high-resolution synchrotron result, showing markedly clearer letters." />
  <figcaption className="mt-0">PHerc. Paris 4: the 2023 Vesuvius Challenge Grand Prize reading (top) and the new 3D ink segmentation (bottom) — the same text, recovered one-to-one from better data.</figcaption>
</figure>

### PHerc. 139 — a title, and an author

In a third scroll, PHerc. 139, we recover the scroll's **title and author attribution**: the work is identified as **Philodemus, _On Gods_, Book 8** — a treatise by the Epicurean philosopher whose works fill so much of this library. Reading the title of a closed scroll tells scholars what a roll contains before a single column of its body is studied.

<figure className="">
  <img src="/img/firstscroll/fig4.webp" alt="PHerc. 139 title region: surface rendering and ink-enhanced view revealing the title and author attribution, identifying the work as Philodemus, On Gods, Book 8." />
  <figcaption className="mt-0">The title region of PHerc. 139. Enhancing the ink signal reveals the title and author attribution, identifying the scroll as Philodemus, <em>On Gods</em>, Book 8.</figcaption>
</figure>

## How it was done

The scans were acquired with high-resolution phase-contrast X-ray microtomography on the BM18 beamline at the [European Synchrotron Radiation Facility (ESRF)](https://www.esrf.fr/) in Grenoble — an instrument able to resolve the wafer-thin, densely packed layers of a Herculaneum roll. The work was carried out in collaboration with the National Library of Naples “Vittorio Emanuele III”, which safeguards the Herculaneum papyri. From those volumes, the team reconstructed the scroll's geometry, traced and flattened its surface into a readable sheet, and trained machine-learning models to detect ink that is almost indistinguishable from the carbonized papyrus beneath it. Each reading was then examined and transcribed by papyrologists.

<figure className="">
  <video autoPlay playsInline loop muted className="w-[100%] rounded-lg" poster="/img/firstscroll/crosscut-poster.webp">
    <source src="/img/firstscroll/crosscut.webm" type="video/webm" />
    <source src="/img/firstscroll/crosscut.mp4" type="video/mp4" />
  </video>
  <figcaption className="mt-0">A cross-section sweeping through the X-ray scan of PHerc. 1667, revealing the sheet of papyrus wound inside the sealed roll.</figcaption>
</figure>

Crucially, all of this is **open**. The tomographic data, reconstructed surfaces and transcriptions are released under a Creative Commons licence at [scrollprize.org/data](/data) and archived at the ESRF, and the code is on [GitHub](https://github.com/ScrollPrize/villa). Anyone can check the work, build on it, and apply it to the scrolls that remain.

## A victory for open, global science

This is what open science makes possible. The virtual unwrapping of the Herculaneum scrolls was pioneered at EduceLab by its principal investigator, Professor Brent Seales. In 2023 Seales opened his lab's imaging and software technology to the Vesuvius Challenge — a public, donation-funded effort he co-founded with Nat Friedman and Daniel Gross to read the scrolls in the open — and from there a global community took up the problem. The [first letters](/firstletters) and the [2023 Grand Prize](/grandprize) were won by contestants from across the world.

What is less widely known is what happened next. **Most of the Vesuvius Challenge research team first arrived as contestants.** They entered the open competition, won prizes for the breakthroughs they made, and were then recruited onto the team that has now read an entire scroll. The people behind this breakthrough are, in large part, the global community the Challenge itself created.

## What's next

PHerc. 1667 is one scroll. Hundreds more remain sealed — an entire library of philosophy, poetry and prose waiting to be read for the first time since antiquity. The method shown here is built to scale, and everything needed to apply it is open.

If you want to help read the rest of the library:

- **Read the science:** the <a href="/pdf/main.pdf" target="_blank" rel="noopener">preprint (PDF)</a>, also on  <a href="https://arxiv.org/abs/2606.29085" target="_blank" rel="noopener">arXiv</a>.
- **Get the data and code:** [scrollprize.org/data](/data) and [GitHub](https://github.com/ScrollPrize/villa).
- **Join the effort:** [get started](/get_started) and become part of the community reading the scrolls.

The thoughts of the ancient world, sealed in darkness for two millennia, are coming back into the light — a whole scroll at a time.
