import React, { useEffect, useRef, useState } from "react";
import useBrokenLinks from "@docusaurus/useBrokenLinks";
import Head from "@docusaurus/Head";
import Heading from "@theme/Heading";
import useDocusaurusContext from "@docusaurus/useDocusaurusContext";
import { usePluginData } from "@docusaurus/useGlobalData";
import BeforeAfter from "./BeforeAfter";
import LatestPosts from "./LatestPosts";
import {
  creators,
  projectLead,
  sponsors,
  team,
  partners,
} from "./landingData";

/* ==========================================================================
   Landing — "Obsidian Minimal" restyle (WP1).
   One card recipe (.vc-card), one accent (--vc-accent), no gradient text,
   no box shadows. Layout/skin lives in src/css/landing.css.
   ========================================================================== */

const usd = new Intl.NumberFormat("en-US", {
  style: "currency",
  currency: "USD",
  maximumFractionDigits: 0,
});

/* ---------------------------------------------------------------------------
   Story content (JSX-heavy, so it stays here; pure data lives in
   landingData.js). One media element per chapter.
--------------------------------------------------------------------------- */

const storyImage = (src, alt) => (
  <img
    src={src}
    alt={alt}
    loading="lazy"
    decoding="async"
    className="vc-media vc-story__media"
  />
);

const stories = ({ unrollVideo }) => [
  {
    date: "79 AD",
    text: "Mount Vesuvius erupts.",
    anchor: "vesuvius",
    description: (
      <>
        <p>
          In Herculaneum, twenty meters of hot mud and ash bury an enormous
          villa once owned by the father-in-law of Julius Caesar. Inside, there
          is a vast library of papyrus scrolls.
        </p>
        <p>
          The scrolls are carbonized by the heat of the volcanic debris. But
          they are also preserved. For centuries, as virtually every ancient
          text exposed to the air decays and disappears, the library of the
          Villa of the Papyri waits underground, intact.
        </p>
      </>
    ),
  },
  {
    date: "1750 AD",
    text: "A farmer discovers the buried villa.",
    description: (
      <>
        <p>
          While digging a well, an Italian farmworker encounters a marble
          pavement. Excavations unearth beautiful statues and frescoes – and
          hundreds of scrolls. Carbonized and ashen, they are extremely
          fragile. But the temptation to open them is great; if read, they
          would significantly increase the corpus of literature we have from
          antiquity.
        </p>
        <p>
          Early attempts to open the scrolls unfortunately destroy many of
          them. A few are painstakingly unrolled by a monk over several
          decades, and they are found to contain philosophical texts written in
          Greek. More than six hundred remain unopened and unreadable.
        </p>
        {storyImage("/img/landing/scroll.webp", "Carbonized Herculaneum scroll")}
      </>
    ),
  },
  {
    date: "2015 AD",
    text: "Dr. Brent Seales pioneers virtual unwrapping.",
    description: (
      <>
        <p>
          Using X-ray tomography and computer vision, a team led by Dr. Brent
          Seales at the University of Kentucky reads the En-Gedi scroll without
          opening it. Discovered in the Dead Sea region of Israel, the scroll
          is found to contain text from the book of Leviticus.
        </p>
        <p>
          Virtual unwrapping has since emerged as a growing field with multiple
          successes. Their work went on to show the elusive carbon ink of the
          Herculaneum scrolls can also be detected using X-ray tomography,
          laying the foundation for Vesuvius Challenge.
        </p>
        <video
          playsInline
          loop
          muted
          preload="metadata"
          title="Virtual unwrapping of the En-Gedi scroll"
          aria-label="Virtual unwrapping of the En-Gedi scroll"
          className="vc-media vc-story__media"
          poster="/img/landing/engedi5.webp"
          ref={unrollVideo}
        >
          <source src="/img/landing/engedi5.webm" type="video/webm" />
        </video>
      </>
    ),
  },
  {
    date: "2023 AD",
    text: "A remarkable breakthrough.",
    description: (
      <>
        <p>
          Vesuvius Challenge launched in March 2023 with a Grand Prize for
          the first team to recover four passages of 140 characters from a
          Herculaneum scroll. Within a year,{" "}
          <a href="/grandprize">the prize was claimed</a>. The quest was
          just beginning.
        </p>
        <div className="vc-media vc-panorama">
          <img
            src="/img/landing/scroll-full-min.webp"
            alt="Herculaneum scroll panorama"
            loading="lazy"
            decoding="async"
          />
        </div>
      </>
    ),
  },
];

/* --------------------------------------------------------------------------
   Small presentational pieces
-------------------------------------------------------------------------- */

const Story = ({ story, index }) => (
  <article
    id={`story-section-${index}`}
    className="vc-story"
    aria-labelledby={story.anchor || `story-title-${index}`}
  >
    <p className="vc-kicker vc-story__kicker">{story.date}</p>
    <Heading
      as="h2"
      id={story.anchor || `story-title-${index}`}
      className="vc-story__title"
    >
      {story.text}
    </Heading>
    <div className="vc-story__body">{story.description}</div>
  </article>
);

/* Open-prize board — data-driven from docs/34_prizes.md frontmatter via
   plugins/prizes-data.js. Dense competition-platform rows: ember amount,
   title + hook, cadence. Whole row links into /prizes. */
const OpenPrizeRow = ({ prize }) => (
  <a
    href={prize.href}
    className={`vc-open-row${prize.featured ? " vc-open-row--featured" : ""}`}
  >
    <span className="vc-open-row__amount vc-nums">
      {usd.format(prize.amount)}
      {prize.unit && (
        <span className="vc-open-row__unit">{prize.unit}</span>
      )}
    </span>
    <span className="vc-open-row__body">
      <span className="vc-open-row__title">{prize.title}</span>
      <span className="vc-open-row__hook">{prize.hook}</span>
      {prize.tiers && (
        <span className="vc-open-row__tiers" aria-label="Prize tiers">
          {prize.tiers.map((tier) => (
            <span className="vc-chip vc-open-row__tier" key={tier.name}>
              {tier.name}{" "}
              <span className="vc-nums vc-open-row__tier-amt">
                {usd.format(tier.amount)}
              </span>
            </span>
          ))}
        </span>
      )}
    </span>
    <span className="vc-open-row__meta">{prize.cadence}</span>
  </a>
);

const OpenPrizeBoard = ({ prizes }) => {
  if (!prizes.length) return null;
  const total = prizes.reduce((sum, p) => sum + p.amount, 0);
  return (
    <div className="vc-card vc-card--flush vc-open-board">
      <div className="vc-open-board__head">
        <span className="vc-label">Currently open</span>
        <span className="vc-open-board__total vc-nums">
          {usd.format(total)} prize pool
        </span>
      </div>
      {prizes.map((p) => (
        <OpenPrizeRow prize={p} key={p.id || p.title} />
      ))}
    </div>
  );
};

/* --------------------------------------------------------------------------
   2027 Grand Prize sticker — $1,000,000 announcement, landing page only.
   Advertises the TOTAL open prize pool (bigger number than any single
   prize), computed from the prizes-page frontmatter; the whole card links
   to /prizes. Enters after a small scroll (or a short delay on desktop,
   where the bottom-center pill can't collide with the hero CTAs); steps
   aside while the footer or the Open Prizes board is on screen; dismissible
   with a 44px close button, persisted in localStorage (SSR-safe: window
   access only inside effects). Toggle with SHOW_PRIZE_STICKER.
-------------------------------------------------------------------------- */
const SHOW_PRIZE_STICKER = true;
const PRIZE_STICKER_DISMISS_KEY = "vcPrizePoolDismissed";

function PrizePoolSticker({ total }) {
  const [shown, setShown] = useState(false);
  const [dismissed, setDismissed] = useState(false);
  const [yielding, setYielding] = useState(false);

  useEffect(() => {
    if (typeof window === "undefined") return undefined;
    try {
      if (window.localStorage.getItem(PRIZE_STICKER_DISMISS_KEY) === "1") {
        setDismissed(true);
        return undefined;
      }
    } catch (e) {
      // Storage unavailable (privacy mode) — still show; dismissal just
      // won't persist across visits.
    }
    let revealed = false;
    let timer = null;
    const onScroll = () => {
      if (window.scrollY > 240) reveal();
    };
    const reveal = () => {
      if (revealed) return;
      revealed = true;
      window.removeEventListener("scroll", onScroll);
      if (timer) window.clearTimeout(timer);
      setShown(true);
    };
    window.addEventListener("scroll", onScroll, { passive: true });
    // Desktop-only delay fallback: on phones the sticker waits for a scroll
    // so it can never pop over the hero CTAs at the bottom of the first view.
    if (window.matchMedia("(min-width: 769px)").matches) {
      timer = window.setTimeout(reveal, 3000);
    }
    onScroll(); // already scrolled (deep link / restored position)
    return () => {
      window.removeEventListener("scroll", onScroll);
      if (timer) window.clearTimeout(timer);
    };
  }, []);

  // Step aside while a footer is visible (links stay uncovered) or while
  // the Open Prizes board is on screen (no point advertising the $1M row
  // the visitor is already looking at — and on phones the bar would cover
  // that very row).
  useEffect(() => {
    if (
      typeof window === "undefined" ||
      typeof IntersectionObserver === "undefined"
    )
      return undefined;
    const watched = Array.from(document.querySelectorAll("footer")).concat(
      Array.from(
        document.querySelectorAll('section[aria-labelledby="open-prizes"]')
      )
    );
    if (!watched.length) return undefined;
    const onScreen = new Set();
    const observer = new IntersectionObserver((entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) onScreen.add(entry.target);
        else onScreen.delete(entry.target);
      });
      setYielding(onScreen.size > 0);
    });
    watched.forEach((el) => observer.observe(el));
    return () => observer.disconnect();
  }, []);

  if (!SHOW_PRIZE_STICKER || dismissed) return null;
  const interactive = shown && !yielding;
  return (
    <div
      className={`vc-gp2027${shown ? " vc-gp2027--in" : ""}${
        yielding ? " vc-gp2027--yield" : ""
      }`}
      role="complementary"
      aria-label="Open prize pool announcement"
      aria-hidden={!interactive}
    >
      <a
        href="/prizes"
        className="vc-gp2027__card"
        tabIndex={interactive ? 0 : -1}
      >
        <span className="vc-label vc-gp2027__kicker">Open prize pool</span>
        <span className="vc-gp2027__amount vc-nums">{usd.format(total)}</span>
        <span className="vc-gp2027__hint">
          From monthly awards to the $1,000,000 Grand Prize&nbsp;→
        </span>
      </a>
      <button
        type="button"
        className="vc-gp2027__close"
        aria-label="Dismiss the open prize pool announcement"
        tabIndex={interactive ? 0 : -1}
        onClick={() => {
          setDismissed(true);
          try {
            window.localStorage.setItem(PRIZE_STICKER_DISMISS_KEY, "1");
          } catch (e) {
            // Best effort — session-only dismissal.
          }
        }}
      >
        ×
      </button>
    </div>
  );
}

const SponsorAvatar = ({ sponsor }) => {
  if (!sponsor.image) {
    return <span className="vc-sponsor__avatar" aria-hidden="true" />;
  }
  const images = Array.isArray(sponsor.image) ? sponsor.image : [sponsor.image];
  return (
    <span className="vc-avatars vc-avatars--mono vc-sponsor__avatar">
      {images.map((img, i) => (
        <img
          key={i}
          src={img}
          alt=""
          loading="lazy"
          decoding="async"
          style={{ zIndex: 10 - i }}
        />
      ))}
    </span>
  );
};

const SponsorRow = ({ sponsor, dense }) => (
  <a
    href={sponsor.href}
    className={`vc-sponsor${dense ? " vc-sponsor--dense" : ""}`}
    target="_blank"
    rel="nofollow sponsored noopener noreferrer"
  >
    {!dense && <SponsorAvatar sponsor={sponsor} />}
    <span className="vc-sponsor__name">{sponsor.name}</span>
    <span className="vc-sponsor__amount vc-nums">
      {usd.format(sponsor.amount)}
    </span>
  </a>
);

/* Sponsor ordering: amount descending; ties alphabetical, Anonymous last. */
const sponsorOrder = (a, b) =>
  b.amount - a.amount ||
  (a.name === "Anonymous") - (b.name === "Anonymous") ||
  a.name.localeCompare(b.name);

const SponsorTier = ({ label, title, list, dense, collapsible }) => {
  const grid = (
    <div className={`vc-tier__grid${dense ? " vc-tier__grid--dense" : ""}`}>
      {list.map((s, i) => (
        <SponsorRow sponsor={s} dense={dense} key={i} />
      ))}
    </div>
  );
  if (collapsible) {
    return (
      <details className="vc-tier vc-collapse">
        <summary className="vc-collapse__summary">
          <span className="vc-collapse__heading">
            <span className="vc-label vc-tier__label">{label}</span>
            <span className="vc-tier__title vc-collapse__title">
              {title}
              <span className="vc-collapse__count"> ({list.length})</span>
            </span>
          </span>
          <span className="vc-collapse__arrow" aria-hidden="true">▾</span>
        </summary>
        <div className="vc-collapse__body">{grid}</div>
      </details>
    );
  }
  return (
    <div className="vc-tier">
      <p className="vc-label vc-tier__label">{label}</p>
      <h3 className="vc-tier__title">{title}</h3>
      {grid}
    </div>
  );
};

/* Team group — plain by default; `collapsible` folds advisor/alumni lists
   behind an expander so the lower page stays short (content preserved). */
const TeamGroup = ({ title, list, collapsible }) => {
  const people = list.map((t, i) => <PersonLink link={t} key={i} />);
  if (collapsible) {
    return (
      <details className="vc-team__group vc-collapse">
        <summary className="vc-collapse__summary">
          <span className="vc-collapse__heading">
            <span className="vc-collapse__title vc-team__group-title">
              {title}
              <span className="vc-collapse__count"> ({list.length})</span>
            </span>
          </span>
          <span className="vc-collapse__arrow" aria-hidden="true">▾</span>
        </summary>
        <div className="vc-collapse__body">{people}</div>
      </details>
    );
  }
  return (
    <div className="vc-team__group">
      <h3>{title}</h3>
      {people}
    </div>
  );
};

const PersonLink = ({ link }) => (
  <div className="vc-person">
    {link.href ? (
      <a href={link.href} className="vc-person__name">
        {link.name}
      </a>
    ) : (
      <span className="vc-person__name">{link.name}</span>
    )}
    {link.title && <span className="vc-person__role">{link.title}</span>}
  </div>
);

const ChallengeBox = ({
  title,
  titleHref,
  children,
  skills,
  linkText,
  href,
  media,
  bounty,
}) => (
  <div className="vc-card vc-problem">
    <div className="vc-problem__text">
      <h3 className="vc-problem__title">
        {titleHref ? (
          <a
            href={titleHref}
            className="text-inherit hover:underline hover:text-inherit"
          >
            {title}
          </a>
        ) : (
          title
        )}
      </h3>
      <div className="vc-problem__body">{children}</div>
      {skills && (
        <div className="vc-chips" aria-label="Related skills">
          {skills.map((skill) => (
            <span className="vc-chip" key={skill}>
              {skill}
            </span>
          ))}
        </div>
      )}
      {bounty && (
        <a href={bounty.href} className="vc-problem__bounty">
          <span className="vc-label">Solve it, win</span>
          <span className="vc-problem__bounty-text">{bounty.text}</span>
        </a>
      )}
      <a href={href} className="vc-cta">
        {linkText}
      </a>
    </div>
    <div className="vc-problem__media">{media}</div>
  </div>
);

const autoPlay = (ref) =>
  ref &&
  ref.current &&
  ref.current
    .play()
    .then(() => {})
    .catch(() => {
      // Video couldn't play; poster stays visible.
    });

/* --------------------------------------------------------------------------
   Page
-------------------------------------------------------------------------- */

export function Landing() {
  const { siteConfig } = useDocusaurusContext();
  const canonicalUrl = `${siteConfig?.url ?? ""}${siteConfig?.baseUrl ?? "/"}`;

  // EMBARGO FLAG — local preview only. Keep false in any online commit/deploy;
  // flip to true on 2026-06-25 (Naples press conference) to launch publicly.
  const SHOW_BREAKING = true;

  useBrokenLinks().collectAnchor("sponsors");
  useBrokenLinks().collectAnchor("partners");
  useBrokenLinks().collectAnchor("our-story");

  // siteUrl is used by the OpenGraph/Twitter tags below. Sitewide JSON-LD
  // (Organization + WebSite) is injected via headTags in docusaurus.config.js
  // so it is present in the server-rendered static HTML (react-helmet drops
  // <script> children from SSR output).
  const siteUrl = (siteConfig?.url ?? "") + (siteConfig?.baseUrl ?? "/");

  // Days until the 2027 Grand Prize deadline (client-only to avoid an SSR
  // hydration mismatch; the chip simply appears after mount).
  const [gpDaysLeft, setGpDaysLeft] = useState(null);
  useEffect(() => {
    const deadline = new Date("2027-06-25T23:59:59-07:00");
    setGpDaysLeft(
      Math.max(0, Math.ceil((deadline.getTime() - Date.now()) / 86400000)),
    );
  }, []);

  const heroVideo = useRef(null);
  const revealVideo = useRef(null);
  const unrollVideo = useRef(null);

  useEffect(() => {
    if (typeof window === "undefined") return;
    // prefers-reduced-motion -> poster only: never attach the deferred hero
    // video source, never autoplay anything.
    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) return;
    // Defer hero video source until idle (improves LCP); poster-first.
    const v = heroVideo.current;
    const srcEl = v?.querySelector("source[data-src]");
    const enableVideo = () => {
      if (!v || !srcEl || srcEl.src) return;
      srcEl.src = srcEl.dataset.src;
      v.load();
      autoPlay(heroVideo);
    };
    // The reveal card's unrolling video: desktop-only (phones keep the 29 KB
    // end-frame poster), data-saver-aware, plays ONCE and holds on its final
    // frame — the fully-read scroll.
    const rv = revealVideo.current;
    const enableReveal = () => {
      if (!rv) return;
      if (!window.matchMedia("(min-width: 997px)").matches) return;
      if (navigator.connection?.saveData) return;
      const sources = rv.querySelectorAll("source[data-src]");
      if (!sources.length || sources[0].src) return;
      // Swap to the first-frame poster so playback is monotonic (rolled →
      // unrolls once → holds); the end-frame poster stays for poster-only
      // contexts (mobile, reduced-motion, data-saver).
      rv.poster = "/img/firstscroll/hero-reveal-start-960.webp";
      sources.forEach((s) => {
        s.src = s.dataset.src;
      });
      rv.load();
      autoPlay(revealVideo);
    };
    const enableAll = () => {
      enableVideo();
      enableReveal();
    };
    if ("requestIdleCallback" in window)
      window.requestIdleCallback(enableAll, { timeout: 1200 });
    else setTimeout(enableAll, 600);
    autoPlay(unrollVideo);
  }, []);

  // Open prizes are sourced from docs/34_prizes.md frontmatter at build time
  // (plugins/prizes-data.js) — the landing updates with the prizes page.
  const { prizes: openPrizes = [] } = usePluginData("prizes-data") || {};
  const { awardedTotal = 0 } = usePluginData("winners-data") || {};
  const { counts = {} } = usePluginData("atlas-data") || {};
  const openPrizeTotal = openPrizes.reduce((sum, p) => sum + p.amount, 0);
  const openById = Object.fromEntries(openPrizes.map((p) => [p.id, p]));
  const grandPrize2027 = openById["grand-prize-2027"];
  const firstLetters = openById["first-letters"];
  const firstTitle = openById["first-title"];
  const progressPrizes = openById["progress-prizes"];

  return (
    <>
      <Head>
        <title>
          Vesuvius Challenge — Reading the Herculaneum Scrolls with AI
        </title>
        <meta
          name="description"
          content={`Vesuvius Challenge uses machine learning and computer vision to read the carbonized Herculaneum scrolls buried by Vesuvius in 79 AD. Over ${usd.format(awardedTotal)} awarded.`}
        />
        <link rel="canonical" href={canonicalUrl} />
        {/* Preload the LCP poster image */}
        <link
          rel="preload"
          as="image"
          href="/img/landing/vesuvius.webp"
          fetchpriority="high"
        />
        {/* Reveal-card poster (desktop hero right column) */}
        <link
          rel="preload"
          as="image"
          href="/img/firstscroll/hero-reveal-end-960.webp"
          media="(min-width: 997px)"
          fetchpriority="high"
        />
        {/* OpenGraph/Twitter */}
        <meta property="og:type" content="website" />
        <meta property="og:url" content={siteUrl} />
        <meta
          property="og:title"
          content="Vesuvius Challenge — Reading the Herculaneum Scrolls with AI"
        />
        <meta
          property="og:description"
          content={`Vesuvius Challenge uses machine learning and computer vision to read the carbonized Herculaneum scrolls buried by Vesuvius in 79 AD. Over ${usd.format(awardedTotal)} awarded.`}
        />
        <meta
          property="og:image"
          content={siteUrl + "img/social/opengraph.jpg"}
        />
        <meta property="og:image:width" content="1200" />
        <meta property="og:image:height" content="630" />
        <meta name="twitter:card" content="summary_large_image" />
        <meta
          name="twitter:title"
          content="Vesuvius Challenge — Reading the Herculaneum Scrolls with AI"
        />
        <meta
          name="twitter:description"
          content={`Vesuvius Challenge uses machine learning and computer vision to read the carbonized Herculaneum scrolls buried by Vesuvius in 79 AD. Over ${usd.format(awardedTotal)} awarded.`}
        />
        <meta
          name="twitter:image"
          content={siteUrl + "img/social/opengraph.jpg"}
        />
        {/* Sitewide JSON-LD (Organization + WebSite) is injected via headTags in docusaurus.config.js */}
      </Head>

      <div className="vc-landing">
        {/* ------------------------------------------------------------------
            Hero — static --vc-bg base; the volcano video is a demoted backdrop
            (≤40% opacity) behind a solid→transparent overlay on the media.
            Height is navbar-aware (never bare 100vh).
        ------------------------------------------------------------------ */}
        <section className="vc-hero" aria-labelledby="home-hero-title">
          <div className="vc-hero__backdrop" aria-hidden="true">
            <video
              playsInline
              loop
              muted
              preload="metadata"
              title="Vesuvius volcano hero background"
              poster="/img/landing/vesuvius.webp"
              className="vc-hero__video"
              ref={heroVideo}
            >
              <source
                data-src="/img/landing/vesuvius-flipped-960.webm"
                type="video/webm"
              />
            </video>
            <div className="vc-hero__scrim" />
          </div>
          <div className="container mx-auto vc-hero__content">
            <div className="vc-hero__grid">
              <div className="vc-hero__main">
                <Heading as="h1" id="home-hero-title" className="vc-hero__title">
                  Resurrect an ancient library from the ashes of a volcano.
                </Heading>
                <p className="vc-hero__tagline">Win Prizes. Make History.</p>
                <p className="vc-hero__intro">
                  Vesuvius Challenge is a machine learning, computer vision, and
                  geometry competition that is <a href="/firstscroll">reading</a>{" "}
                  the carbonized Herculaneum scrolls without opening them.
                </p>
                <div className="vc-hero__ctas">
                  <a className="vc-btn" href="/get_started">
                    Get Started
                  </a>
                  <a
                    className="vc-btn-outline"
                    href="https://discord.gg/V4fJhvtaQn"
                  >
                    Join Discord
                  </a>
                </div>
              </div>

              <div className="vc-hero__aside">
                {gpDaysLeft !== null && (
                  <a
                    className="vc-hero__countdown"
                    href="/prizes#2027-grand-prize"
                    aria-label={`${gpDaysLeft} days to the $1,000,000 Grand Prize deadline`}
                  >
                    <span className="vc-hero__countdown-label">
                      $1,000,000 Grand Prize
                    </span>
                    <span className="vc-hero__countdown-row">
                      <span className="vc-hero__countdown-value vc-nums">
                        {gpDaysLeft}
                      </span>
                      <span className="vc-hero__countdown-unit">
                        days
                        <br />
                        left
                      </span>
                    </span>
                    <span className="vc-hero__countdown-date">
                      deadline June 25th, 2027
                    </span>
                  </a>
                )}
              </div>
            </div>

            <div className="vc-stat-strip vc-hero__stats">
                {openPrizes.length > 0 && (
                  <a className="vc-stat vc-stat--link" href="#open-prizes">
                    <span className="vc-stat__value">
                      {usd.format(openPrizeTotal)}
                    </span>
                    <span className="vc-stat__label">open prize pool</span>
                  </a>
                )}
                {awardedTotal > 0 && (
                  <a className="vc-stat vc-stat--link" href="/winners">
                    <span className="vc-stat__value">
                      {usd.format(awardedTotal)}
                    </span>
                    <span className="vc-stat__label">already awarded</span>
                  </a>
                )}
                <a className="vc-stat vc-stat--link" href="/data_browser">
                  <span className="vc-stat__value">
                    {(counts.scrolls ?? 35) + (counts.fragments ?? 10)}
                  </span>
                  <span className="vc-stat__label">scrolls &amp; fragments scanned</span>
                </a>
                <a className="vc-stat vc-stat--link" href="/data_browser">
                  <span className="vc-stat__value">1</span>
                  <span className="vc-stat__label">scroll fully read</span>
                </a>
              </div>
          </div>
        </section>

        {/* News teasers (LatestPosts internals are compressed by WP2) */}
        <section
          className="vc-section vc-section--tight"
          aria-label="Latest updates"
        >
          <div className="container mx-auto">
            <a
              className="vc-updates__head"
              href="https://scrollprize.substack.com"
            >
              News — from our Substack
            </a>
            <div className="vc-newsbar">
              {/* The unwrap payoff: PHerc. 1667 unrolling into readable Greek.
                  Poster = the video's final frame, so poster-only contexts
                  (mobile, reduced-motion, data-saver) still get the reveal. */}
              {SHOW_BREAKING && (
                <a href="/firstscroll" className="vc-newsbar__feature">
                  <video
                    playsInline
                    muted
                    preload="none"
                    title="PHerc. 1667 virtually unwrapping into readable Greek"
                    poster="/img/firstscroll/hero-reveal-end-960.webp"
                    className="vc-newsbar__media"
                    ref={revealVideo}
                  >
                    <source
                      data-src="/img/firstscroll/hero-reveal.webm"
                      type="video/webm"
                    />
                    <source
                      data-src="/img/firstscroll/hero-reveal.mp4"
                      type="video/mp4"
                    />
                  </video>
                  <span className="vc-newsbar__feature-title">
                    We read an entire scroll&nbsp;→
                  </span>
                </a>
              )}
              <LatestPosts />
            </div>
          </div>
        </section>

        {/* ------------------------------------------------------------------
            Open problems — two before/after sliders (Virtual Unwrapping,
            Ink Detection) + "Targets" strip (the merged "What We're
            Building Towards").
        ------------------------------------------------------------------ */}
        <section className="vc-section" aria-labelledby="open-problems">
          <div className="container mx-auto">
            <Heading as="h2" id="open-problems" className="vc-h2">
              Open problems
            </Heading>
            <div className="vc-problems">
              <ChallengeBox
                title="Virtual Unwrapping"
                titleHref="/2026_open_problems#2-unwrapping-turning-disconnected-voxels-into-a-surface"
                linkText="Current Path"
                href="/tutorial_spiral"
                bounty={
                  (grandPrize2027 || progressPrizes) && {
                    href: (grandPrize2027 || progressPrizes).href,
                    text: (
                      <>
                        {grandPrize2027 && (
                          <>
                            <strong className="vc-nums">
                              {usd.format(grandPrize2027.amount)}
                            </strong>{" "}
                            {grandPrize2027.title}
                          </>
                        )}
                        {grandPrize2027 && progressPrizes && " · "}
                        {progressPrizes && (
                          <>
                            <strong className="vc-nums">
                              {usd.format(progressPrizes.amount)}
                            </strong>{" "}
                            {progressPrizes.unit} in {progressPrizes.title}
                          </>
                        )}
                        &nbsp;→
                      </>
                    ),
                  }
                }
                skills={[
                  "geometry processing",
                  "3D computer vision",
                  "optimization",
                  "C++",
                ]}
                media={
                  <BeforeAfter
                    beforeImage="/img/data/raw_pred.png"
                    afterImage="/img/data/patches.png"
                  />
                }
              >
                <p>
                  A CT scan yields voxels, not columns: the writing surface
                  must be segmented, meshed, and flattened. The pipeline
                  fails where adjacent sheets are densely packed, or tear.
                  Tracing remains semi-automated. Fully automating it is an
                  open problem.
                </p>
              </ChallengeBox>

              <ChallengeBox
                title="Ink Detection"
                titleHref="/2026_open_problems#3-ink-recovery-reading-the-scrolls"
                linkText="Find a Letter"
                href="/tutorial5"
                bounty={
                  firstLetters &&
                  firstTitle && {
                    href: firstLetters.href,
                    text: (
                      <>
                        <strong className="vc-nums">
                          {usd.format(firstLetters.amount)}
                        </strong>{" "}
                        {firstLetters.title} ·{" "}
                        <strong className="vc-nums">
                          {usd.format(firstTitle.amount)}
                        </strong>{" "}
                        {firstTitle.title}&nbsp;→
                      </>
                    ),
                  }
                }
                skills={[
                  "machine learning",
                  "computer vision",
                  "domain generalization",
                ]}
                media={
                  <BeforeAfter
                    beforeImage="/img/ink/51002_crop/32.jpg"
                    afterImage="/img/ink/51002_crop/prediction.jpg"
                  />
                }
              >
                <p>
                  Carbon ink is nearly indistinguishable from papyrus in
                  X-ray CT. Models train on fragments with visible ink and
                  infer it inside sealed scrolls; iterative pseudo-labeling
                  bootstraps legibility. Ink has surfaced on 9 of the 45 scanned
                  scrolls and fragments, not always legibly. Generalization is the open
                  problem.
                </p>
              </ChallengeBox>
            </div>
          </div>
        </section>

        {/* ------------------------------------------------------------------
            Open prizes — dense board fed by the prizes page frontmatter
            (plugins/prizes-data.js). Sits directly above Open problems:
            the money first, then the problems it pays for.
        ------------------------------------------------------------------ */}
        <section className="vc-section" aria-labelledby="open-prizes">
          <div className="container mx-auto">
            <Heading as="h2" id="open-prizes" className="vc-h2">
              Open prizes
            </Heading>
            <OpenPrizeBoard prizes={openPrizes} />
            <a href="/prizes" className="vc-cta">
              Rules and details
            </a>
          </div>
        </section>


        {/* ------------------------------------------------------------------
            Our story — five beats on one hairline timeline. The 2026 chapter
            carries the open prizes (merged per audit §4).
        ------------------------------------------------------------------ */}
        <section className="vc-section" aria-labelledby="our-story">
          <div className="container mx-auto">
            <Heading as="h2" id="our-story" className="vc-h2">
              Our story
            </Heading>
            <div className="vc-timeline">
              {(() => {
                const allStories = stories({ unrollVideo });
                const EARLY_COUNT = 3; // 79 AD, 1750, 2015 — the pre-Challenge backstory
                const early = allStories.slice(0, EARLY_COUNT);
                const recent = allStories.slice(EARLY_COUNT);
                return (
                  <>
                    <details className="vc-collapse vc-story-history">
                      <summary className="vc-collapse__summary">
                        <span className="vc-collapse__heading">
                          <span className="vc-collapse__title">
                            The backstory
                          </span>
                          <span className="vc-collapse__hint">
                            79 AD – 2015 AD · how we got here
                          </span>
                        </span>
                        <span className="vc-collapse__arrow" aria-hidden="true">
                          ▾
                        </span>
                      </summary>
                      <div className="vc-collapse__body">
                        {early.map((s, index) => (
                          <Story story={s} key={s.date} index={index} />
                        ))}
                      </div>
                    </details>
                    {recent.map((s, index) => (
                      <Story
                        story={s}
                        key={s.date}
                        index={index + EARLY_COUNT}
                      />
                    ))}
                  </>
                );
              })()}

              <article
                className="vc-story"
                aria-labelledby="the-challenge-continues"
              >
                <p className="vc-kicker vc-story__kicker">2026 AD</p>
                <Heading
                  as="h2"
                  id="the-challenge-continues"
                  className="vc-story__title"
                >
                  The first scroll is read.
                </Heading>
                <div className="vc-story__body">
                  <p>
                    In 2026, <a href="/firstscroll">PHerc. 1667</a> became
                    the first Herculaneum scroll to be virtually unwrapped and
                    read end to end. The challenge now moves onto its next
                    stage: reading multiple entire scrolls.
                  </p>
                  <div className="vc-media">
                    <BeforeAfter
                      beforeImage="/img/firstscroll/hero-reveal-start-960-crop.webp"
                      afterImage="/img/firstscroll/hero-reveal-end-960-crop.webp"
                      altBefore="PHerc. 1667 as a sealed CT scan, before virtual unwrapping"
                      altAfter="The unwrapped writing surface of PHerc. 1667, showing columns of ancient Greek text"
                      className="aspect-[20/7] h-auto"
                    />
                  </div>
                  <a href="/prizes" className="vc-cta">
                    See all open prizes
                  </a>
                </div>
              </article>
            </div>
          </div>
        </section>

        {/* ------------------------------------------------------------------
            Sponsors — every donor, neutral tier headings, dense rows.
        ------------------------------------------------------------------ */}
        <section className="vc-section" aria-labelledby="sponsors">
          <div className="container mx-auto">
            <div className="vc-h2-row">
              <Heading as="h2" id="sponsors" className="vc-h2">
                Sponsors
              </Heading>
              <a
                href="https://donate.stripe.com/aEUg101vt9eN8gM144"
                className="vc-btn-outline vc-h2-row__action"
              >
                Donate
              </a>
            </div>
            <SponsorTier
              label="$200,000 and above"
              title="Caesars"
              list={sponsors
                .filter((s) => s.amount >= 200000)
                .sort(sponsorOrder)}
            />
            {/* Senators + Citizens expanders sit side by side on desktop
                (vertical-space save); they stack on phones. */}
            <div className="vc-tier-row">
              <SponsorTier
                label="$50,000 – $200,000"
                title="Senators"
                list={sponsors
                  .filter((s) => s.amount >= 50000 && s.amount < 200000)
                  .sort(sponsorOrder)}
                collapsible
              />
              <SponsorTier
                label="Up to $50,000"
                title="Citizens"
                list={sponsors
                  .filter((s) => s.amount < 50000)
                  .sort(sponsorOrder)}
                dense
                collapsible
              />
            </div>
          </div>
        </section>

        {/* ------------------------------------------------------------------
            Team — one credit line + the plain dense lists (already the target
            aesthetic; heading scale fixed).
        ------------------------------------------------------------------ */}
        <section className="vc-section" aria-labelledby="team">
          <div className="container mx-auto">
            <Heading as="h2" id="team" className="vc-h2">
              Team
            </Heading>
            <div className="vc-credits">
              <div className="vc-credits__group">
                <p className="vc-label vc-credits__label">Created by</p>
                {creators.map((c) => (
                  <div className="vc-person" key={c.name}>
                    <a href={c.href} className="vc-person__name">
                      {c.name}
                    </a>
                    {c.role && (
                      <span className="vc-person__role">{c.role}</span>
                    )}
                  </div>
                ))}
              </div>
              <div className="vc-credits__group">
                <p className="vc-label vc-credits__label">Led by</p>
                <div className="vc-person">
                  <a href={projectLead.href} className="vc-person__name">
                    {projectLead.name}
                  </a>
                  <span className="vc-person__role">{projectLead.title}</span>
                </div>
              </div>
            </div>
            <div className="vc-team">
              <TeamGroup title="Tech Team" list={team.challenge} />
              <TeamGroup title="Papyrology Team" list={team.papyrology} />
              <TeamGroup title="Annotation Team" list={team.annotation} />
              <TeamGroup
                title="EduceLab Team (Partners)"
                list={team.educe}
                collapsible
              />
              <TeamGroup title="Advisors & Alumni" list={team.alumni} collapsible />
              <TeamGroup
                title="Papyrology Advisors"
                list={team.papyrologyAdvisors}
                collapsible
              />
            </div>
          </div>
        </section>

        {/* ------------------------------------------------------------------
            Partners — one monochrome logo row + EduceLab funders.
        ------------------------------------------------------------------ */}
        <section
          className="vc-section vc-section--last"
          aria-labelledby="partners"
        >
          <div className="container mx-auto">
            <Heading as="h2" id="partners" className="vc-h2">
              Partners
            </Heading>
            <div className="vc-partners">
              {partners.map((p, i) => (
                <a key={i} href={p.href} className="vc-partners__link">
                  <img
                    src={p.icon}
                    alt={p.name}
                    loading="lazy"
                    decoding="async"
                    className={`vc-partners__img${
                      p.tall ? " vc-partners__img--tall" : ""
                    }`}
                  />
                </a>
              ))}
            </div>
          </div>
        </section>

        {/* $1,000,000 · 2027 Grand Prize floating teaser (landing only). */}
        <PrizePoolSticker total={openPrizeTotal} />
      </div>
    </>
  );
}
