import React, { useEffect, useRef } from "react";
import useBrokenLinks from "@docusaurus/useBrokenLinks";
import Head from "@docusaurus/Head";
import Heading from "@theme/Heading";
import useDocusaurusContext from "@docusaurus/useDocusaurusContext";
import BeforeAfter from "./BeforeAfter";
import LatestPosts from "./LatestPosts";
import {
  prizes,
  creators,
  sponsors,
  team,
  partners,
  educelabFunders,
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
   landingData.js). One media element per chapter — except 2024, which keeps
   its two-result pair (per audit §4).
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
        {storyImage(
          "/img/landing/rocio-espin-pinar-villa-papyri-small.webp",
          "Villa of the Papyri illustration"
        )}
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
          Vesuvius Challenge was launched in March 2023 to bring the world
          together to read the Herculaneum scrolls. Along with smaller progress
          prizes, a Grand Prize was issued for the first team to recover 4
          passages of 140 characters from a Herculaneum scroll.
        </p>
        <p>
          Following a year of remarkable progress,{" "}
          <a href="/grandprize">the prize was claimed</a>. After 275 years, the
          ancient puzzle of the Herculaneum Papyri has been cracked open. But
          the quest to uncover the secrets of the scrolls is just beginning.
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
  {
    date: "2024 AD",
    text: "New frontiers.",
    description: (
      <>
        <p>
          A widespread community effort builds on the success of the first
          scroll, automating and refining the components of the virtual
          unwrapping pipeline. Efforts to scan and read multiple scrolls are
          underway. New text is revealed from another scroll.
        </p>
        <div className="vc-story__pair">
          <img
            src="/img/landing/patches.webp"
            alt="Automatically segmented papyrus surface patches"
            loading="lazy"
            decoding="async"
            className="vc-media"
          />
          <img
            src="/img/landing/scroll5.webp"
            alt="New text revealed from another scroll"
            loading="lazy"
            decoding="async"
            className="vc-media"
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

const Winners = ({ winners }) => (
  <span className="vc-avatars">
    {winners.map((winner, i) => (
      <img
        key={i}
        src={winner.image}
        alt={winner.name}
        loading="lazy"
        decoding="async"
        style={{ zIndex: 10 - i }}
      />
    ))}
  </span>
);

const OpenPrize = ({ prize }) => (
  <a href={prize.href} className="vc-card vc-open-prize">
    <span className="vc-open-prize__amount vc-nums">{prize.prizeMoney}</span>
    <span className="vc-open-prize__title">{prize.title}</span>
    <span className="vc-open-prize__desc">{prize.description}</span>
  </a>
);

const AwardedPrizeRow = ({ prize }) => (
  <a href={prize.href} className="vc-prize-row">
    <span className="vc-prize-row__title">{prize.title}</span>
    <span className="vc-prize-row__winners">
      <Winners winners={prize.winners} />
      <span className="vc-prize-row__label">
        {prize.winnersLabel || `${prize.winners.length} Winners`}
      </span>
    </span>
    <span className="vc-prize-row__amount vc-nums">{prize.prizeMoney}</span>
  </a>
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

const SponsorTier = ({ label, title, list, dense }) => (
  <div className="vc-tier">
    <p className="vc-label vc-tier__label">{label}</p>
    <h3 className="vc-tier__title">{title}</h3>
    <div className={`vc-tier__grid${dense ? " vc-tier__grid--dense" : ""}`}>
      {list.map((s, i) => (
        <SponsorRow sponsor={s} dense={dense} key={i} />
      ))}
    </div>
  </div>
);

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

const ChallengeBox = ({ title, children, skills, linkText, href, media }) => (
  <div className="vc-card vc-problem">
    <div className="vc-problem__text">
      <h3 className="vc-problem__title">{title}</h3>
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
      <a href={href} className="vc-cta">
        {linkText}
      </a>
    </div>
    <div className="vc-problem__media">{media}</div>
  </div>
);

const targets = [
  {
    title: "Accurate Surface Representation",
    description:
      "We lack the accuracy to make the meshing step as simple as it could be.",
  },
  {
    title: "Generalizable Ink Detection",
    description:
      "Ink has been found in two scrolls, but remains elusive in our other scrolls.",
  },
  {
    title: "High Quality Annotations",
    description: "We need an abundance of high-quality annotations.",
  },
  {
    title: "Robust Meshing",
    description:
      "Methods that function where Surface Representation is unreliable are needed.",
  },
];

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
  useBrokenLinks().collectAnchor("educelab-funders");
  useBrokenLinks().collectAnchor("our-story");

  // siteUrl is used by the OpenGraph/Twitter tags below. Sitewide JSON-LD
  // (Organization + WebSite) is injected via headTags in docusaurus.config.js
  // so it is present in the server-rendered static HTML (react-helmet drops
  // <script> children from SSR output).
  const siteUrl = (siteConfig?.url ?? "") + (siteConfig?.baseUrl ?? "/");

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

  const openPrizes = prizes.filter((p) => !p.winners);
  const awardedPrizes = prizes.filter((p) => p.winners);

  return (
    <>
      <Head>
        <title>
          Vesuvius Challenge — Reading the Herculaneum Scrolls with AI
        </title>
        <meta
          name="description"
          content="Vesuvius Challenge uses machine learning and computer vision to read the carbonized Herculaneum scrolls buried by Vesuvius in 79 AD. Over $1,800,500 awarded."
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
          content="Vesuvius Challenge uses machine learning and computer vision to read the carbonized Herculaneum scrolls buried by Vesuvius in 79 AD. Over $1,800,500 awarded."
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
          content="Vesuvius Challenge uses machine learning and computer vision to read the carbonized Herculaneum scrolls buried by Vesuvius in 79 AD. Over $1,800,500 awarded."
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
            <Heading as="h1" id="home-hero-title" className="vc-hero__title">
              Resurrect an ancient library from the ashes of a volcano.
            </Heading>
            <p className="vc-hero__tagline">Win Prizes. Make History.</p>
            <p className="vc-hero__intro">
              Vesuvius Challenge is a machine learning, computer vision, and
              geometry competition that is <a href="/grandprize">reading</a>{" "}
              the carbonized Herculaneum scrolls — without opening them. Our
              current challenge: <a href="/get_started">join the community</a>{" "}
              and grow from a few passages to entire scrolls.
            </p>
            <div className="vc-stat-strip vc-hero__stats">
              <div className="vc-stat">
                <span className="vc-stat__value">$1,800,500</span>
                <span className="vc-stat__label">awarded in prizes</span>
              </div>
              <div className="vc-stat">
                <span className="vc-stat__value">5</span>
                <span className="vc-stat__label">scrolls scanned</span>
              </div>
              <div className="vc-stat">
                <span className="vc-stat__value">2</span>
                <span className="vc-stat__label">scrolls read</span>
              </div>
            </div>
            <div className="vc-hero__ctas">
              <a className="vc-btn" href="/get_started">
                Get Started
              </a>
              <a className="vc-cta" href="/firstscroll">
                Read the breaking announcement
              </a>
            </div>

            {SHOW_BREAKING && (
              <a
                href="/firstscroll"
                className="vc-card vc-card--hero vc-breaking"
              >
                <span className="vc-breaking__text">
                  <span className="vc-kicker">Breaking</span>
                  <span className="vc-breaking__title">
                    We read an entire Herculaneum scroll.
                  </span>
                  <span className="vc-breaking__desc">
                    PHerc. 1667, sealed since 79&nbsp;AD, has been virtually
                    unwrapped and read end to end. Read the announcement&nbsp;→
                  </span>
                </span>
                <img
                  src="/img/firstscroll/banner-strip.webp"
                  alt="The unwrapped writing surface of PHerc. 1667, columns of ancient Greek."
                  decoding="async"
                  className="vc-media vc-breaking__img"
                />
              </a>
            )}
          </div>
        </section>

        {/* News teasers (LatestPosts internals are compressed by WP2) */}
        <section
          className="vc-section vc-section--tight"
          aria-label="Latest updates"
        >
          <div className="container mx-auto">
            <LatestPosts />
          </div>
        </section>

        {/* ------------------------------------------------------------------
            Open problems — three before/after sliders + "Targets" strip
            (the merged "What We're Building Towards").
        ------------------------------------------------------------------ */}
        <section className="vc-section" aria-labelledby="open-problems">
          <div className="container mx-auto">
            <Heading as="h2" id="open-problems" className="vc-h2">
              Open problems
            </Heading>
            <div className="vc-problems">
              <ChallengeBox
                title="Representation"
                linkText="Scan the Surface"
                href="/unwrapping"
                skills={[
                  "image annotation",
                  "computer vision",
                  "machine learning",
                  "medical imaging",
                ]}
                media={
                  <BeforeAfter
                    beforeImage="/img/data/rep_raw_10037.png"
                    afterImage="/img/data/rep_norms_10037.png"
                  />
                }
              >
                <p>
                  Carbonized and crushed under pyroclastic flow and debris, the
                  scrolls are in rough shape. Tracing the 3D sheets through
                  these damaged scrolls is nearly impossible in the raw scan
                  data. More structured representations, like those obtained
                  with semantic segmentation, simplify downstream tasks
                  significantly.
                </p>
              </ChallengeBox>

              <ChallengeBox
                title="Geometric Reconstruction"
                linkText="Chart the Path"
                href="/segmentation"
                skills={[
                  "geometry processing",
                  "computer vision",
                  "machine learning",
                  "optimization",
                ]}
                media={
                  <BeforeAfter
                    beforeImage="/img/data/raw_pred.png"
                    afterImage="/img/data/patches.png"
                  />
                }
              >
                <p>
                  A better image representation alone does not an unrolled
                  scroll make. We need methods to better map the surfaces,
                  stitch them where necessary, and extract them into readable
                  sheets of papyrus. For a primer on current autosegmentation
                  methods and their progress, read the{" "}
                  <a href="/unwrapping">Virtual Unwrapping document</a>.
                </p>
              </ChallengeBox>

              <ChallengeBox
                title="Ink Detection"
                linkText="Find a Letter"
                href="/tutorial5"
                skills={[
                  "image annotation",
                  "computer vision",
                  "machine learning",
                  "pattern recognition",
                ]}
                media={
                  <BeforeAfter
                    beforeImage="/img/ink/51002_crop/32.jpg"
                    afterImage="/img/ink/51002_crop/prediction.jpg"
                  />
                }
              >
                <p>
                  We've so far recovered text from just two of our five
                  scrolls. Is the ink fundamentally different in others? Is the
                  papyrus surface? We're not yet sure. We are certain though
                  that if it ever existed, it can be detected.
                </p>
              </ChallengeBox>
            </div>

            <div className="vc-targets">
              <p className="vc-label vc-targets__label">
                What we're building towards
              </p>
              <div className="vc-targets__grid">
                {targets.map((t) => (
                  <div className="vc-target" key={t.title}>
                    <span className="vc-target__title">{t.title}</span>
                    <span className="vc-target__desc">{t.description}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </section>

        {/* ------------------------------------------------------------------
            Our story — six beats on one hairline timeline. The 2026 chapter
            carries the open prizes (merged per audit §4).
        ------------------------------------------------------------------ */}
        <section className="vc-section" aria-labelledby="our-story">
          <div className="container mx-auto">
            <Heading as="h2" id="our-story" className="vc-h2">
              Our story
            </Heading>
            <div className="vc-timeline">
              {stories({ unrollVideo }).map((s, index) => (
                <Story story={s} key={s.date} index={index} />
              ))}

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
                  The challenge continues.
                </Heading>
                <div className="vc-story__body">
                  <p>
                    Vesuvius Challenge moves onto its next stage of reading
                    multiple entire scrolls. Read more about the prizes below,
                    and on how they contribute towards{" "}
                    <a href="/master_plan">The Master Plan</a>.
                  </p>
                  <div className="vc-open-prizes">
                    {openPrizes.map((p) => (
                      <OpenPrize prize={p} key={p.title} />
                    ))}
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
            Awarded prizes — dense rows: name / winners / $ (tabular).
        ------------------------------------------------------------------ */}
        <section className="vc-section" aria-labelledby="awarded-prizes">
          <div className="container mx-auto">
            <Heading as="h2" id="awarded-prizes" className="vc-h2">
              Awarded prizes
            </Heading>
            <p className="vc-section__intro">
              Incredible teams of engineers are helping us unlock these
              secrets, providing unprecedented access to scrolls that have not
              been read in two millennia.
            </p>
            <div className="vc-card vc-card--flush vc-prize-table">
              {awardedPrizes.map((p) => (
                <AwardedPrizeRow prize={p} key={p.title} />
              ))}
            </div>
            <a href="/winners" className="vc-cta vc-prize-table__cta">
              All winners
            </a>
          </div>
        </section>

        {/* ------------------------------------------------------------------
            Sponsors — every donor, neutral tier headings, dense rows.
        ------------------------------------------------------------------ */}
        <section className="vc-section" aria-labelledby="sponsors">
          <div className="container mx-auto">
            <Heading as="h2" id="sponsors" className="vc-h2">
              Sponsors
            </Heading>
            <SponsorTier
              label="$200,000 and above"
              title="Caesars"
              list={sponsors.filter((s) => s.amount >= 200000)}
            />
            <SponsorTier
              label="$50,000 – $200,000"
              title="Senators"
              list={sponsors.filter(
                (s) => s.amount >= 50000 && s.amount < 200000
              )}
            />
            <SponsorTier
              label="Up to $50,000"
              title="Citizens"
              list={sponsors.filter((s) => s.amount < 50000)}
              dense
            />
            <div className="vc-sponsors__cta">
              <a
                href="https://donate.stripe.com/aEUg101vt9eN8gM144"
                className="vc-btn-outline"
              >
                Become a sponsor
              </a>
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
            <p className="vc-credit">
              Created by{" "}
              {creators.map((c, i) => (
                <React.Fragment key={c.name}>
                  {i > 0 && (i === creators.length - 1 ? ", and " : ", ")}
                  <a href={c.href}>{c.name}</a>
                </React.Fragment>
              ))}
              .
            </p>
            <div className="vc-team">
              <div className="vc-team__group">
                <h3>Vesuvius Challenge Team</h3>
                {team.challenge.map((t, i) => (
                  <PersonLink link={t} key={i} />
                ))}
              </div>
              <div className="vc-team__group">
                <h3>EduceLab Team</h3>
                {team.educe.map((t, i) => (
                  <PersonLink link={t} key={i} />
                ))}
              </div>
              <div className="vc-team__group">
                <h3>Advisors & Alumni</h3>
                {team.alumni.map((t, i) => (
                  <PersonLink link={t} key={i} />
                ))}
              </div>
              <div className="vc-team__group">
                <h3>Papyrology Team</h3>
                {team.papyrology.map((t, i) => (
                  <PersonLink link={t} key={i} />
                ))}
              </div>
              <div className="vc-team__group">
                <h3>Papyrology Advisors</h3>
                {team.papyrologyAdvisors.map((t, i) => (
                  <PersonLink link={t} key={i} />
                ))}
              </div>
            </div>
            <p className="vc-caption vc-team__credit">
              Villa dei Papiri art by{" "}
              <a href="https://www.artstation.com/rocioespin">Rocío Espín</a>
            </p>
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
            <h3 className="vc-funders__title" id="educelab-funders">
              EduceLab funders
            </h3>
            <div className="vc-funders">
              {educelabFunders.map((t, i) => (
                <PersonLink link={t} key={i} />
              ))}
            </div>
          </div>
        </section>
      </div>
    </>
  );
}
