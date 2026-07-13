// @ts-check
// Note: type annotations allow type checking and IDEs autocompletion

const lightCodeTheme = require("prism-react-renderer").themes.github;
const darkCodeTheme = require("prism-react-renderer").themes.vsDark;
const rehypeKatex = require("rehype-katex").default; // Extract default export for rehype-katex
const remarkMath = require("remark-math").default; // Extract default export for remark-math
const rehypeImageDimensions = require("./plugins/rehype-image-dimensions");
const { computeAwardedTotal } = require("./plugins/winners-data");

// Total $ awarded, summed from the winners page's money headings at config
// load — keeps the tagline/meta descriptions in sync with docs/15_winners.md.
const AWARDED = computeAwardedTotal(__dirname).total.toLocaleString("en-US");

// Sitewide structured data. Injected via headTags (below) so it is present in
// the server-rendered static HTML — react-helmet (<Head>) drops <script>
// children from SSR output, which is why prior on-page JSON-LD was invisible
// to non-JS crawlers.
const SITE_URL = "https://scrollprize.org/";
const orgJsonLd = {
  "@context": "https://schema.org",
  "@type": "Organization",
  name: "Vesuvius Challenge",
  alternateName: "Scroll Prize",
  url: SITE_URL,
  logo: SITE_URL + "img/social/opengraph.jpg",
  description:
    `Vesuvius Challenge is a machine learning, computer vision, and geometry competition reading the carbonized Herculaneum scrolls — papyri buried by the eruption of Mount Vesuvius in 79 AD — using X-ray CT scanning and AI. Over $${AWARDED} in prizes has been awarded.`,
  foundingDate: "2023",
  founder: [
    { "@type": "Person", name: "Nat Friedman" },
    { "@type": "Person", name: "Daniel Gross" },
    { "@type": "Person", name: "Brent Seales" },
  ],
  employee: {
    "@type": "Person",
    name: "Giorgio Angelotti",
    jobTitle: "Project Lead",
  },
  sameAs: [
    "https://en.wikipedia.org/wiki/Vesuvius_Challenge",
    "https://x.com/scrollprize",
    "https://github.com/ScrollPrize",
    "https://scrollprize.substack.com",
    "https://www.youtube.com/@scrollprize",
  ],
};
const webSiteJsonLd = {
  "@context": "https://schema.org",
  "@type": "WebSite",
  name: "Vesuvius Challenge",
  url: SITE_URL,
  publisher: { "@type": "Organization", name: "Vesuvius Challenge" },
};

// The sitemap's per-page <lastmod> is derived from git history. Some build
// environments have no git repository available — notably Vercel monorepo
// subdir builds, which run in a copied directory with no .git — and there the
// git lookup throws and aborts the whole build. Detect git up-front and only
// enable lastmod when it's actually available; otherwise omit it (the prior
// behavior) so the build always succeeds.
function gitLastmodSupported() {
  try {
    require("child_process").execSync("git rev-parse --is-inside-work-tree", {
      cwd: __dirname,
      stdio: "ignore",
    });
    return true;
  } catch (e) {
    return false;
  }
}
const SITEMAP_LASTMOD = gitLastmodSupported() ? "date" : null;

/** @type {import('@docusaurus/types').Config} */
const config = {
  title: "Vesuvius Challenge",
  tagline: `A machine learning and computer vision competition with $${AWARDED} awarded in prizes`,
  url: "https://scrollprize.org",
  baseUrl: "/",
  onBrokenAnchors: "throw",
  onBrokenLinks: "throw",
  onBrokenMarkdownLinks: "throw",
  favicon: "img/social/favicon.ico",
  customFields: {
    // "Ask the Scrolls" chat endpoint. Same-origin Vercel function in prod;
    // point CHAT_ENDPOINT at a local dev shim (scripts/devChatServer.mjs) when
    // testing against `yarn serve`. Empty string disables the widget.
    chatEndpoint:
      process.env.CHAT_ENDPOINT !== undefined
        ? process.env.CHAT_ENDPOINT
        : "/api/chat",
  },
  scripts: [
    {
      src: "https://cdn.usefathom.com/script.js",
      "data-site": "XERDEBQR",
      defer: true,
      "data-spa": "auto",
    },
  ],
  headTags: [
    {
      tagName: "meta",
      attributes: { name: "theme-color", content: "#000000" },
    },
    {
      tagName: "link",
      attributes: { rel: "apple-touch-icon", href: "/apple-touch-icon.png" },
    },
    {
      tagName: "link",
      attributes: { rel: "manifest", href: "/manifest.json" },
    },
    {
      tagName: "script",
      attributes: { type: "application/ld+json" },
      innerHTML: JSON.stringify(orgJsonLd),
    },
    {
      tagName: "script",
      attributes: { type: "application/ld+json" },
      innerHTML: JSON.stringify(webSiteJsonLd),
    },
  ],
  clientModules: [
    require.resolve("./src/clientModules/imageZoom.js"),
    require.resolve("./src/clientModules/gtagSafeStub.js"),
    require.resolve("./src/clientModules/anchorScrollFix.js"),
  ],
  markdown: {
    mermaid: true,
  },
  themes: ["@docusaurus/theme-mermaid"],

  i18n: {
    defaultLocale: "en",
    locales: ["en"],
  },

  presets: [
    [
      "classic",
      /** @type {import('@docusaurus/preset-classic').Options} */
      ({
        docs: {
          routeBasePath: "/",
          sidebarPath: require.resolve("./sidebars.js"),
          sidebarCollapsible: true,
          breadcrumbs: false,
          remarkPlugins: [remarkMath],
          rehypePlugins: [[rehypeKatex, { strict: false }], rehypeImageDimensions],
        },
        blog: false,
        theme: {
          customCss: [
            require.resolve("./src/css/tokens.css"),
            require.resolve("./src/css/utilities.css"),
            require.resolve("./src/css/custom.css"),
            require.resolve("./src/css/chrome.css"),
            require.resolve("./src/css/landing.css"),
            require.resolve("./src/css/getstarted.css"),
            require.resolve("./src/css/imageZoom.css"),
            require.resolve("./src/css/chat.css"),
          ],
        },
        gtag: {
          trackingID: "G-NLQQENBL0L",
          anonymizeIP: false,
        },
        sitemap: {
          lastmod: SITEMAP_LASTMOD,
          changefreq: null,
          priority: null,
          filename: 'sitemap.xml',
          ignorePatterns: [],
        },
      }),
    ],
  ],

  themeConfig:
  /** @type {import('@docusaurus/preset-classic').ThemeConfig} */
      ({
        navbar: {
          title: "Vesuvius Challenge",
          logo: {
            alt: "Vesuvius Challenge Logo",
            src: "img/social/favicon-64x64.png",
          },
          items: [
            { to: "/prizes", label: "Prizes", position: "left" },
            { to: "/2026_open_problems", label: "Problems", position: "left" },
            {
              type: "dropdown",
              label: "Tutorials & In-depth",
              position: "left",
              items: [
                {
                  to: "/tutorial_VC3D",
                  label: "Virtual Unwrapping",
                },
                {
                  to: "/tutorial_spiral",
                  label: "Spiral Fitting",
                },
                {
                  to: "/open_problems/winding_annotations",
                  label: "Winding Constraints",
                },
                {
                  to: "/tutorial5",
                  label: "Ink Detection",
                },
              ],
            },
            { to: "/data_browser", label: "Data", position: "left" },
            { to: "/winners", label: "Milestones", position: "left" },
            {
              type: "dropdown",
              label: "Community",
              position: "left",
              items: [
                { label: "Discord", href: "https://discord.gg/V4fJhvtaQn" },
                { label: "𝕏", href: "https://x.com/scrollprize" },
                { label: "Substack", href: "https://scrollprize.substack.com" },
                { label: "GitHub", href: "https://github.com/ScrollPrize" },
                {
                  label: "Hugging Face",
                  href: "https://huggingface.co/scrollprize",
                },
                {
                  label: "Weights & Biases",
                  href: "https://wandb.ai/vesuvius-challenge/projects",
                },
                {
                  label: "Donate",
                  href: "https://donate.stripe.com/aEUg101vt9eN8gM144",
                },
              ],
            },
            {
              href: "https://discord.gg/V4fJhvtaQn",
              label: "Join Discord",
              position: "right",
              className: "vc-navbar-discord",
            },
            {
              to: "/get_started",
              label: "Get Started",
              position: "right",
              className: "vc-navbar-cta",
            },
          ],
        },
        footer: {
          style: "dark",
          links: [
            {
              items: [
                {
                  label: "Discord",
                  href: "https://discord.gg/V4fJhvtaQn",
                },
                {
                  label: "GitHub",
                  href: "https://github.com/ScrollPrize",
                },
                {
                  label: "Hugging Face",
                  href: "https://huggingface.co/scrollprize",
                },
                {
                  label: "Weights & Biases",
                  href: "https://wandb.ai/vesuvius-challenge/projects",
                },
                {
                  label: "Substack",
                  href: "https://scrollprize.substack.com",
                },
                {
                  label: "𝕏",
                  href: "https://x.com/scrollprize",
                },
                {
                  label: "Jobs",
                  to: "/jobs",
                },
              ],
            },
          ],
          copyright: `Copyright © ${new Date().getFullYear()} Vesuvius Challenge · Content licensed CC BY-NC 4.0`,
        },
        image: '/img/social/opengraph.jpg',
        metadata: [
          {
            name: "description",
            content: `A machine learning and computer vision competition with $${AWARDED} awarded in prizes`,
          },
          {
            property: "og:type",
            content: "website",
          },
          {
            property: "og:url",
            content: "https://scrollprize.org/",
          },
          {
            property: "og:title",
            content: "Vesuvius Challenge",
          },
          {
            property: "og:description",
            content: "A machine learning & computer vision competition.",
          },
          {
            property: "og:image",
            content: "https://scrollprize.org/img/social/opengraph.jpg",
          },
          {
            property: "twitter:card",
            content: "summary_large_image",
          },
          {
            property: "twitter:url",
            content: "https://scrollprize.org/",
          },
          {
            property: "twitter:title",
            content: "Vesuvius Challenge",
          },
          {
            property: "twitter:description",
            content: "A machine learning & computer vision competition.",
          },
          {
            property: "twitter:image",
            content: "https://scrollprize.org/img/social/opengraph.jpg",
          },
        ],
        prism: {
          theme: lightCodeTheme,
          darkTheme: darkCodeTheme,
        },
        colorMode: {
          defaultMode: "dark",
          disableSwitch: true,
          respectPrefersColorScheme: false,
        },
      }),

  plugins: [
    async function myPlugin(context, options) {
      return {
        name: "docusaurus-tailwindcss",
        configurePostCss(postcssOptions) {
          postcssOptions.plugins.push(require("tailwindcss"));
          if (process.env.NODE_ENV !== "development") {
            postcssOptions.plugins.push(require("autoprefixer"));
          }
          return postcssOptions;
        },
      };
    },
    './plugins/fetch-substack-posts',
    './plugins/atlas-data',
    './plugins/prizes-data',
    './plugins/winners-data',
    [
      "@docusaurus/plugin-client-redirects",
      {
        redirects: [
          {
            to: "https://donate.stripe.com/aEUg101vt9eN8gM144",
            from: "/donate",
          },
          {
            to: "/villa_model",
            from: "/lego",
          },
          {
            to: "/unwrapping",
            from: "/unrolling",
          },
          {
            to: "https://dl.ash2txt.org/LICENSE.txt",
            from: "/license",
          },
          // Orphan docs removed in the "Obsidian Minimal" restyle. Journals and
          // Discord threads have linked these for years — keep the URLs alive.
          {
            to: "/segmentation",
            from: "/tutorial_thaumato",
          },
          {
            to: "/segmentation",
            from: "/tutorial4",
          },
          {
            to: "/unwrapping",
            from: "/open_problem_rep",
          },
          // The open_problems/ink_detection page was retired (superseded by
          // the Ink recovery section of Problems in-depth + the ink
          // tutorial) — keep the URL alive.
          {
            to: "/tutorial5",
            from: "/open_problems/ink_detection",
          },
          // Pages retired in the C6 fix round — keep the URLs alive.
          {
            to: "/data",
            from: "/data_segments",
          },
          {
            to: "/data",
            from: "/data_fragments",
          },
          {
            to: "/",
            from: "/background",
          },
          // The blogpost briefly lived at /from_ash_to_text and /tech_blogpost
          // on preview builds.
          {
            to: "/2026_open_problems",
            from: "/from_ash_to_text",
          },
          {
            to: "/2026_open_problems",
            from: "/tech_blogpost",
          },
        ],
      },
    ],
  ],
};

module.exports = config;
