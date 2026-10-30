import React from "react";
import { usePluginData } from "@docusaurus/useGlobalData";
import "../css/press.css";

/* Press page pieces: who to contact, live key facts, and the media kit. */

const usd = new Intl.NumberFormat("en-US", {
  style: "currency",
  currency: "USD",
  maximumFractionDigits: 0,
});

const contacts = [
  {
    need: "Press & media, use of images",
    who: "press@scrollprize.org",
    href: "mailto:press@scrollprize.org",
  },
  {
    need: "Technology & AI",
    who: "Paul Henderson",
    role: "Technical Lead",
    href: "mailto:paul@scrollprize.org",
  },
  {
    need: "Papyrology, archaeology & history",
    who: "Federica Nicolardi",
    role: "Papyrology Lead",
    href: "https://www.docenti.unina.it/federica.nicolardi",
  },
];

export const PressContacts = () => (
  <dl className="vc-press-contacts">
    {contacts.map((c) => (
      <div className="vc-press-contacts__row" key={c.need}>
        <dt className="vc-label">{c.need}</dt>
        <dd>
          <a href={c.href}>{c.who}</a>
          {c.role && <span className="vc-press-contacts__role">{c.role}</span>}
        </dd>
      </div>
    ))}
  </dl>
);

/* Counts come from the same build-time data as the landing page
   (plugins/winners-data.js, plugins/atlas-data.js), so they stay current;
   a fact whose data is missing is left out rather than shown as zero. */
export const PressFacts = () => {
  const { awardedTotal = 0 } = usePluginData("winners-data") || {};
  const { counts = {} } = usePluginData("atlas-data") || {};
  const facts = [
    { value: "2023", label: "Vesuvius Challenge launched" },
    awardedTotal > 0 && {
      value: usd.format(awardedTotal),
      label: "awarded in prizes so far",
    },
    counts.scrolls > 0 && {
      value: `${counts.scrolls + (counts.fragments || 0)}`,
      label: `scrolls & fragments scanned, data open to all`,
    },
    { value: "2026", label: "first Herculaneum scroll read end to end" },
  ].filter(Boolean);
  return (
    <div className="vc-press-facts">
      {facts.map((f) => (
        <div className="vc-press-facts__item" key={f.label}>
          <span className="vc-press-facts__value vc-nums">{f.value}</span>
          <span className="vc-press-facts__label">{f.label}</span>
        </div>
      ))}
    </div>
  );
};

const media = [
  {
    thumb: "/img/firstscroll/hero-poster.webp",
    file: "/img/firstscroll/hero-reveal.mp4",
    format: "MP4 · 3.3 MB",
    caption: "PHerc. 1667 being virtually unwrapped.",
  },
  {
    thumb: "/img/firstscroll/banner-strip.webp",
    file: "/img/firstscroll/banner-full.webp",
    format: "WebP · 12000 × 924",
    caption:
      "The complete unwrapped writing surface of PHerc. 1667: roughly 1.4 metres of papyrus and around twenty-two columns of Greek.",
  },
  {
    thumb: "/img/firstscroll/fig1.webp",
    file: "/img/firstscroll/fig1.webp",
    format: "WebP · 2400 × 2148",
    caption:
      "From object to text: the sealed roll, cross-sections through its X-ray scan, and the unwrapped surface of PHerc. 1667.",
  },
  {
    thumb: "/img/firstscroll/crosscut-poster.webp",
    file: "/img/firstscroll/crosscut.mp4",
    format: "MP4 · 4.9 MB",
    caption:
      "A cross-section sweeping through the X-ray scan of PHerc. 1667, revealing the sheet of papyrus wound inside the sealed roll.",
  },
  {
    thumb: "/img/firstscroll/fig3.webp",
    file: "/img/firstscroll/fig3.webp",
    format: "WebP · 2400 × 1750",
    caption:
      "PHerc. Paris 4: the 2023 Grand Prize reading (top) and the newer 3D ink segmentation (bottom) of the same text.",
  },
  {
    thumb: "/img/firstscroll/fig4.webp",
    file: "/img/firstscroll/fig4.webp",
    format: "WebP · 2400 × 1862",
    caption:
      "The title of PHerc. 139, revealed by enhancing the ink: Philodemus, On Gods, Book 8.",
  },
];

export const PressMedia = () => (
  <div className="vc-press-media">
    {media.map((m) => (
      <figure className="vc-press-media__item" key={m.file}>
        <a href={m.file} target="_blank" rel="noopener">
          <img
            src={m.thumb}
            alt={m.caption}
            loading="lazy"
            decoding="async"
            className="vc-press-media__thumb"
          />
        </a>
        <figcaption>
          {m.caption}
          <a href={m.file} className="vc-press-media__download" download>
            Download <span className="vc-nums">({m.format})</span>
          </a>
        </figcaption>
      </figure>
    ))}
  </div>
);
