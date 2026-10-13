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
  projectLeads,
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
        <p className="text-sm opacity-60 mt-2">
          © EduceLab/University of Kentucky.
        </p>
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
        <p className="text-sm opacity-60 mt-2">
          Virtual unwrapping of the En-Gedi scroll. © EduceLab/University of Kentucky.
        </p>
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

const SponsorRow = ({ sponsor, hidden }) => (
  <a
    href={sponsor.href}
    className="vc-sponsor"
    target="_blank"
    rel="nofollow sponsored noopener noreferrer"
    hidden={hidden}
  >
    <SponsorAvatar sponsor={sponsor} />
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

/* Every sponsor in one list; gifts below the fold amount stay in the markup
   but hidden until "Show more" is clicked. */
const SPONSOR_FOLD_AMOUNT = 50000;

const SponsorList = ({ list }) => {
  const [expanded, setExpanded] = useState(false);
  const folded = list.filter((s) => s.amount < SPONSOR_FOLD_AMOUNT).length;
  return (
    <>
      <div className="vc-sponsors__grid">
        {list.map((s, i) => (
          <SponsorRow
            sponsor={s}
            hidden={!expanded && s.amount < SPONSOR_FOLD_AMOUNT}
            key={i}
          />
        ))}
      </div>
      {!expanded && folded > 0 && (
        <button
          type="button"
          className="vc-btn-outline vc-sponsors__more"
          onClick={() => setExpanded(true)}
        >
          Show more ({folded})
        </button>
      )}
    </>
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

/* Open problem card: before/after slider, title, short pitch, then the
   prizes that pay for solving it (amounts from docs/34_prizes.md
   frontmatter). */
const ProblemCard = ({
  title,
  titleHref,
  beforeImage,
  afterImage,
  children,
  prizes,
  linkText,
  href,
}) => (
  <div className="vc-card vc-card--flush vc-problem">
    <BeforeAfter
      beforeImage={beforeImage}
      afterImage={afterImage}
      className="vc-problem__slider"
      hint
    />
    <div className="vc-problem__text">
      <h3 className="vc-problem__title">
        <a href={titleHref}>{title}</a>
      </h3>
      <p className="vc-problem__body">
        <a href={titleHref}>{children}</a>
      </p>
      <div className="vc-problem__foot">
        <div className="vc-problem__prizes">
          {prizes.map((p) => (
            <a href={p.href} className="vc-problem__prize" key={p.id}>
              <strong className="vc-nums">{usd.format(p.amount)}</strong>{" "}
              {p.title}
            </a>
          ))}
        </div>
        <a href={href} className="vc-cta">
          {linkText}
        </a>
      </div>
    </div>
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

  // The two "Led by" names swap order at random on each page load so
  // neither is billed first. The server render keeps data order; the coin
  // flip runs after mount to avoid a hydration mismatch.
  const [leads, setLeads] = useState(projectLeads);
  useEffect(() => {
    if (Math.random() < 0.5) setLeads([...projectLeads].reverse());
  }, []);

  const heroVideo = useRef(null);
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
    if ("requestIdleCallback" in window)
      window.requestIdleCallback(enableVideo, { timeout: 1200 });
    else setTimeout(enableVideo, 600);
    autoPlay(unrollVideo);
  }, []);

  // Open prizes are sourced from docs/34_prizes.md frontmatter at build time
  // (plugins/prizes-data.js) — the landing updates with the prizes page.
  const { prizes: openPrizes = [] } = usePluginData("prizes-data") || {};
  const { awardedTotal = 0 } = usePluginData("winners-data") || {};
  const { counts = {} } = usePluginData("atlas-data") || {};
  const openById = Object.fromEntries(openPrizes.map((p) => [p.id, p]));
  const grandPrize2027 = openById["grand-prize-2027"];
  const firstLetters = openById["first-letters-2027"];

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
                <p className="vc-hero__intro">
                  Vesuvius Challenge is an open science research organization{" "}
                  <a href="/firstscroll">reading</a> the carbonized Herculaneum
                  scrolls without opening them. Anyone can help: we run a{" "}
                  <a href="/prizes">$2M competition</a>, with prizes every
                  month.
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
          </div>
        </section>

        {/* News — latest Substack posts, text only */}
        <section
          className="vc-section vc-section--tight"
          aria-labelledby="news"
        >
          <div className="container mx-auto">
            <div className="vc-h2-row vc-updates__row">
              <Heading as="h2" id="news" className="vc-h2">
                News
              </Heading>
              <a
                href="https://scrollprize.substack.com"
                className="vc-cta vc-h2-row__action"
              >
                More on Substack
              </a>
            </div>
            <LatestPosts />
          </div>
        </section>

        {/* ------------------------------------------------------------------
            Open problems — one card per problem, each carrying the prizes
            that pay for it; progress prizes and the full rules below.
        ------------------------------------------------------------------ */}
        <section className="vc-section" aria-labelledby="open-problems">
          <div className="container mx-auto">
            <div className="vc-h2-row">
              <Heading as="h2" id="open-problems" className="vc-h2">
                Open problems &amp; prizes
              </Heading>
              <a href="/prizes" className="vc-cta vc-h2-row__action">
                All prizes and rules
              </a>
            </div>
            <div className="vc-problems">
              <ProblemCard
                title="Virtual Unwrapping"
                titleHref="/2026_open_problems#2-unwrapping-turning-disconnected-voxels-into-a-surface"
                beforeImage="/img/data/raw_pred.png"
                afterImage="/img/data/patches.png"
                prizes={[
                  grandPrize2027 && { ...grandPrize2027, title: "Grand Prize" },
                ].filter(Boolean)}
                linkText="Current Path"
                href="/tutorial_spiral"
              >
                We can trace a scroll's surface, but where sheets are crushed
                or torn it still needs human help. Help us make it fully
                automatic.
              </ProblemCard>

              <ProblemCard
                title="Ink Detection"
                titleHref="/2026_open_problems#3-ink-recovery-reading-the-scrolls"
                beforeImage="/img/ink/51002_crop/32.jpg"
                afterImage="/img/ink/51002_crop/prediction.jpg"
                prizes={[
                  firstLetters && { ...firstLetters, title: "First Letters Prizes" },
                ].filter(Boolean)}
                linkText="Find a Letter"
                href="/tutorial5"
              >
                We can see ink in some scrolls, but models trained on one
                rarely transfer to the next. Help us make them work on every
                scroll.
              </ProblemCard>
            </div>
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
            Sponsors — every donor in one list, smaller gifts behind "Show more".
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
            <SponsorList list={[...sponsors].sort(sponsorOrder)} />
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
                <p className="vc-label vc-credits__label">Led by</p>
                {leads.map((l) => (
                  <div className="vc-person" key={l.name}>
                    <a href={l.href} className="vc-person__name">
                      {l.name}
                    </a>
                    <span className="vc-person__role">{l.title}</span>
                  </div>
                ))}
              </div>
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
            </div>
            <div className="vc-team">
              <TeamGroup title="Technical Team" list={team.challenge} />
              <TeamGroup title="Papyrology Team" list={team.papyrology} />
              <TeamGroup title="Annotation Team" list={team.annotation} />
            </div>
            {/* Collapsible groups sit full width below the columns, so opening
                one never reflows the groups above between columns. */}
            <div className="vc-team__more">
              <TeamGroup title="Advisors & Alumni" list={team.alumni} collapsible />
              <TeamGroup
                title="Papyrology Advisors"
                list={team.papyrologyAdvisors}
                collapsible
              />
              <TeamGroup
                title="EduceLab Team (Partners)"
                list={team.educe}
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
      </div>
    </>
  );
}
