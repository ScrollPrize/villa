// @ts-check
// Note: type annotations allow type checking and IDEs autocompletion

const lightCodeTheme = require("prism-react-renderer").themes.github;
const darkCodeTheme = require("prism-react-renderer").themes.dracula;
const rehypeKatex = require("rehype-katex").default; // Extract default export for rehype-katex
const remarkMath = require("remark-math").default; // Extract default export for remark-math

/** @type {import('@docusaurus/types').Config} */
const config = {
  title: "Vesuvius Challenge",
  tagline: "A $1,000,000+ machine learning and computer vision competition",
  url: "https://scrollprize.org",
  baseUrl: "/",
  onBrokenAnchors: "throw",
  onBrokenLinks: "throw",
  onBrokenMarkdownLinks: "throw",
  favicon: "img/social/favicon.ico",
  scripts: [
    {
      src: "https://cdn.usefathom.com/script.js",
      "data-site": "XERDEBQR",
      defer: true,
      "data-spa": "auto",
    },
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
          sidebarCollapsible: false,
          breadcrumbs: false,
          editUrl:
              "https://github.com/ScrollPrize/villa/tree/main/scrollprize.org",
          remarkPlugins: [remarkMath],
          rehypePlugins: [[rehypeKatex, { strict: false }]],
        },
        blog: false,
        theme: {
          customCss: require.resolve("./src/css/custom.css"),
        },
        gtag: {
          trackingID: "G-NLQQENBL0L",
          anonymizeIP: false,
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
          items: [],
        },
        footer: {
          style: "dark",
          links: [
            {
              title: "Overview",
              items: [
                {
                  label: "Getting Started",
                  to: "/get_started",
                },
                {
                  label: "Master Plan",
                  to: "/master_plan",
                },
              ],
            },
            {
              title: "Community",
              items: [
                {
                  label: "Discord",
                  href: "https://discord.gg/V4fJhvtaQn",
                },
                {
                  label: "Substack",
                  href: "https://scrollprize.substack.com",
                },
                {
                  label: "GitHub",
                  href: "https://github.com/ScrollPrize/villa",
                },
                {
                  label: "𝕏",
                  href: "https://x.com/scrollprize",
                },
              ],
            },
            {
              title: "More",
              items: [
                {
                  label: "Jobs",
                  to: "/jobs",
                },
              ],
            },
          ],
          copyright: `Copyright © ${new Date().getFullYear()} Vesuvius Challenge.`,
        },
        metadata: [
          {
            name: "description",
            content:
                "A $1,000,000+ machine learning and computer vision competition",
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
        ],
      },
    ],
  ],
};

module.exports = config;
