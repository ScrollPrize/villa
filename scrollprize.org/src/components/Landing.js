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
   No boxed sections: content sits on the page, divided by hairlines; only
   media and the hero countdown are framed. One accent (--vc-accent), no
   gradient text, no box shadows. Layout/skin lives in src/css/landing.css.
   ========================================================================== */

const usd = new Intl.NumberFormat("en-US", {
  style: "currency",
  currency: "USD",
  maximumFractionDigits: 0,
});

/* ---------------------------------------------------------------------------
   Our story beats: `line` is the headline sentence, `more` the follow-up,
   `media` one image or clip per beat.
--------------------------------------------------------------------------- */

const storyImage = (src, alt) => (
  <img src={src} alt={alt} loading="lazy" decoding="async" className="vc-media" />
);

/* Muted looping clip; plays on mount unless the viewer prefers reduced
   motion, in which case the poster stays. */
const StoryVideo = ({ sources, poster, title }) => {
  const ref = useRef(null);
  useEffect(() => {
    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) return;
    autoPlay(ref);
  }, []);
  return (
    <video
      ref={ref}
      playsInline
      loop
      muted
      preload="metadata"
      poster={poster}
      title={title}
      aria-label={title}
      className="vc-media"
    >
      {sources.map(([src, type]) => (
        <source src={src} type={type} key={src} />
      ))}
    </video>
  );
};

const storyBeats = [
  {
    date: "79 AD",
    id: "vesuvius",
    line: (
      <>
        Vesuvius erupts, burying a villa and its library of papyrus scrolls
        under twenty meters of ash.
      </>
    ),
    more: (
      <>
        The scrolls are carbonized by the heat, but also preserved. For
        centuries, as almost every other ancient text decays, the library of
        the Villa of the Papyri waits underground, intact.
      </>
    ),
    media: storyImage("/img/landing/vesuvius.webp", "Mount Vesuvius"),
  },
  {
    date: "1750",
    line: (
      <>
        The villa is rediscovered. Early attempts to open the scrolls destroy
        many; hundreds remain unopened.
      </>
    ),
    more: (
      <>
        A farmworker digging a well strikes a marble pavement. Excavations
        unearth hundreds of scrolls, too fragile to unroll. More than six
        hundred remain unread.
      </>
    ),
    media: storyImage("/img/landing/scroll.webp", "Carbonized Herculaneum scroll"),
    credit: true,
  },
  {
    date: "2015",
    line: (
      <>
        Dr. Brent Seales's team virtually unwraps the En-Gedi scroll with X-ray
        tomography.
      </>
    ),
    more: (
      <>
        Their work goes on to show that the elusive carbon ink of the
        Herculaneum scrolls can also be detected, laying the foundation for
        Vesuvius Challenge.
      </>
    ),
    media: (
      <StoryVideo
        sources={[["/img/landing/engedi5.webm", "video/webm"]]}
        poster="/img/landing/engedi5.webp"
        title="Virtual unwrapping of the En-Gedi scroll"
      />
    ),
    credit: true,
  },
  {
    date: "2023",
    line: (
      <>
        Vesuvius Challenge launches; within a year,{" "}
        <a href="/grandprize">the Grand Prize is claimed</a>.
      </>
    ),
    more: (
      <>
        Teams recover four passages of 140 characters from a sealed scroll,
        the first text ever read from inside one.
      </>
    ),
    media: (
      <div className="vc-strip__pan">
        <img
          src="/img/landing/scroll-full-min.webp"
          alt="Herculaneum scroll panorama"
          loading="lazy"
          decoding="async"
        />
      </div>
    ),
  },
  {
    date: "2026",
    line: (
      <>
        PHerc. 1667 becomes the first Herculaneum
        scroll <a href="/firstscroll">read end to end</a>.
      </>
    ),
    more: (
      <>
        Hundreds more remain sealed, and reading them needs unwrapping that is
        faster and more automated. That's the{" "}
        <a href="/prizes#2027-grand-prize">$1M Grand Prize</a>.
      </>
    ),
    media: (
      <StoryVideo
        sources={[
          ["/img/firstscroll/hero-reveal.webm", "video/webm"],
          ["/img/firstscroll/hero-reveal.mp4", "video/mp4"],
        ]}
        poster="/img/firstscroll/hero-poster.webp"
        title="PHerc. 1667 being virtually unwrapped"
      />
    ),
  },
];

const StoryCredit = () => (
  <p className="vc-strip__credit">
    © EduceLab/University of Kentucky
  </p>
);

/* Timeline strip: dates ticked off the header rule, one shared text + media panel.
   Opens on 79 AD so the eruption is the first thing shown. */
const StoryStrip = () => {
  const [active, setActive] = useState(0);
  const beat = storyBeats[active];
  return (
    <div className="vc-strip">
      <div className="vc-strip__dates" role="tablist" aria-label="Timeline">
        {storyBeats.map((b, i) => (
          <button
            type="button"
            role="tab"
            aria-selected={i === active}
            className={`vc-strip__date vc-nums${
              i === active ? " is-active" : ""
            }`}
            onClick={() => setActive(i)}
            key={b.date}
          >
            {b.date}
          </button>
        ))}
      </div>
      <div className="vc-strip__stage">
        <button
          type="button"
          className="vc-strip__arrow"
          aria-label="Previous"
          disabled={active === 0}
          onClick={() => setActive(active - 1)}
        >
          ‹
        </button>
        <div className="vc-strip__panel" role="tabpanel">
          <div className="vc-strip__text">
            <p className="vc-strip__line">{beat.line}</p>
            <p className="vc-strip__more">{beat.more}</p>
          </div>
          <div className="vc-strip__media" key={beat.date}>
            {beat.media}
            {beat.credit && <StoryCredit />}
          </div>
        </div>
        <button
          type="button"
          className="vc-strip__arrow"
          aria-label="Next"
          disabled={active === storyBeats.length - 1}
          onClick={() => setActive(active + 1)}
        >
          ›
        </button>
      </div>
    </div>
  );
};

/* --------------------------------------------------------------------------
   Small presentational pieces
-------------------------------------------------------------------------- */

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
          className="vc-team-tabs__tab vc-sponsors__more"
          aria-expanded={false}
          onClick={() => setExpanded(true)}
        >
          Show more
          <span className="vc-team-tabs__arrow" aria-hidden="true">
            ▾
          </span>
        </button>
      )}
    </>
  );
};

const TeamGroup = ({ title, list }) => (
  <div className="vc-team__group">
    <h3>{title}</h3>
    {list.map((t, i) => (
      <PersonLink link={t} key={i} />
    ))}
  </div>
);

/* Secondary team lists as tabs: at most one open, its people flowing across
   the full width below the tab row. All closed by default to keep the page
   short; clicking the open tab closes it. */
const TeamTabs = ({ groups }) => {
  const [active, setActive] = useState(null);
  return (
    <div className="vc-team-tabs">
      <div
        className={`vc-team-tabs__row${active !== null ? " has-open" : ""}`}
        role="tablist"
      >
        {groups.map((g, i) => (
          <button
            type="button"
            role="tab"
            id={`team-tab-${i}`}
            aria-selected={active === i}
            aria-controls={`team-panel-${i}`}
            className={`vc-team-tabs__tab${active === i ? " is-active" : ""}`}
            onClick={() => setActive(active === i ? null : i)}
            key={g.title}
          >
            {g.title}
            <span className="vc-team-tabs__arrow" aria-hidden="true">
              ▾
            </span>
          </button>
        ))}
      </div>
      {groups.map((g, i) => (
        <div
          role="tabpanel"
          id={`team-panel-${i}`}
          aria-labelledby={`team-tab-${i}`}
          className="vc-team-tabs__panel"
          hidden={active !== i}
          key={g.title}
        >
          {g.list.map((t, k) => (
            <PersonLink link={t} key={k} />
          ))}
        </div>
      ))}
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

/* Open problem column: framed before/after slider, title, short pitch, then the
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
  <div className="vc-problem">
    <BeforeAfter
      beforeImage={beforeImage}
      afterImage={afterImage}
      className="vc-problem__slider"
      hint
    />
    <div className="vc-problem__text">
      <a href={titleHref} className="vc-problem__head">
        <h3 className="vc-problem__title">{title}</h3>
        <p className="vc-problem__body">{children}</p>
      </a>
      <div className="vc-problem__foot">
        <div className="vc-problem__prizes">
          {prizes.map((p) => (
            <p className="vc-problem__line" key={p.id}>
              <a href={p.href} className="vc-problem__prize">
                Win the{" "}
                <strong className="vc-nums">{usd.format(p.amount)}</strong>{" "}
                {p.title}
              </a>
            </p>
          ))}
        </div>
        <p className="vc-problem__line">
          <a href={href} className="vc-problem__link">
            {linkText}
          </a>
        </p>
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
                  Resurrecting an ancient library from the ashes of a volcano.
                </Heading>
                <p className="vc-hero__tagline">
                  Buried in 79 AD, now being revealed.
                </p>
                <p className="vc-hero__intro">
                  Vesuvius Challenge is an open science research organization{" "}
                  <a href="/firstscroll">reading</a> the carbonized Herculaneum
                  scrolls without opening them. <strong>Anyone can help</strong>:
                  we run a{" "}
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
                linkText="Learn about the current path"
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
                linkText="Learn how to find a letter"
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
            Our story — 79 AD to 2026 on one timeline strip; the 2026 beat
            links on to the open prizes.
        ------------------------------------------------------------------ */}
        <section className="vc-section" aria-labelledby="our-story">
          <div className="container mx-auto">
            <div className="vc-h2-row">
              <Heading as="h2" id="our-story" className="vc-h2">
                Our story
              </Heading>
              <a href="/winners" className="vc-cta vc-h2-row__action">
                All milestones
              </a>
            </div>
            <StoryStrip />
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
            {/* Secondary lists as tabs below the columns; the open one spans
                the full width. */}
            <TeamTabs
              groups={[
                { title: "Advisors & Alumni", list: team.alumni },
                { title: "Papyrology Advisors", list: team.papyrologyAdvisors },
                { title: "EduceLab Team (Partners)", list: team.educe },
              ]}
            />
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
                className="vc-cta vc-h2-row__action"
              >
                Make a donation
              </a>
            </div>
            <SponsorList list={[...sponsors].sort(sponsorOrder)} />
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
