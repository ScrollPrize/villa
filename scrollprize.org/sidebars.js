/**
 * Creating a sidebar enables you to:
 - create an ordered group of docs
 - render a sidebar for each doc of that group
 - provide next/previous navigation

 The sidebars can be generated from the filesystem, or explicitly defined here.

 Create as many sidebars as you want.
 */

// @ts-check

const brandHeader = {
  type: 'html',
  value:
    '<a class="navbar__brand custom-top-header" href="/"><div class="navbar__logo"><img src="/img/social/favicon-64x64.png" alt="Vesuvius Challenge Logo" class="themedImage_ToTc themedImage--dark_i4oU"></div><b class="navbar__title text--truncate">Vesuvius Challenge</b></a>',
};

/** @type {import('@docusaurus/plugin-content-docs').SidebarsConfig} */
const sidebars = {
  // Single, unified menu used across the whole site so every important page is
  // reachable from one coherent navigation surface.
  overviewSidebar: [
    brandHeader,
    {
      type: 'category',
      label: 'Overview',
      link: { type: 'doc', id: 'landing' },
      items: [
        { type: 'doc', id: 'get_started' },
        { type: 'doc', id: 'prizes' },
        { type: 'doc', id: 'unwrapping' },
        { type: 'doc', id: 'master_plan' },
      ],
    },
    {
      type: 'category',
      label: 'Data',
      collapsible: true,
      collapsed: true,
      link: { type: 'doc', id: 'data' },
      items: [
        { type: 'doc', id: 'data_browser', label: 'Data Browser' },
        { type: 'doc', id: 'data_segments' },
        { type: 'doc', id: 'data_datasets' },
        { type: 'doc', id: 'data_fragments' },
      ],
    },
    {
      type: 'category',
      label: 'Tutorials',
      collapsible: true,
      collapsed: true,
      link: { type: 'doc', id: 'tutorial' },
      items: [
        { type: 'doc', id: 'tutorial1' },
        { type: 'doc', id: 'tutorial2' },
        { type: 'doc', id: 'tutorial3' },
        { type: 'doc', id: 'tutorial5' },
        { type: 'doc', id: 'segmentation' },
        { type: 'doc', id: 'ink_detection' },
        { type: 'doc', id: 'tutorial_VC' },
      ],
    },
    { type: 'doc', id: 'faq' },
    {
      type: 'category',
      label: 'Milestones & Results',
      link: { type: 'doc', id: 'winners' },
      items: [
        { type: 'doc', id: 'firstscroll' },
        { type: 'doc', id: 'grandprize' },
        { type: 'doc', id: 'grand_prize' },
        { type: 'doc', id: 'firstletters' },
        { type: 'doc', id: 'community_projects' },
      ],
    },
    {
      type: 'category',
      label: 'The Scrolls',
      link: { type: 'doc', id: 'background' },
      items: [
        { type: 'doc', id: 'villa_model' },
        { type: 'doc', id: 'livestream' },
      ],
    },
    {
      type: 'link',
      label: 'Discord',
      href: 'https://discord.gg/V4fJhvtaQn',
    },
    {
      type: 'link',
      label: 'Mailing list',
      href: 'https://scrollprize.substack.com/',
    },
    {
      type: 'link',
      label: '𝕏',
      href: 'https://x.com/scrollprize',
    },
    {
      type: 'link',
      label: 'Donate',
      href: 'https://donate.stripe.com/aEUg101vt9eN8gM144',
    },
    { type: 'doc', id: 'jobs' },
    {
      type: 'category',
      label: 'Archive',
      collapsible: true,
      collapsed: true,
      items: [
        { type: 'doc', id: '28_2024_prizes' },
        { type: 'doc', id: '30_2024_gp_submissions' },
        { type: 'doc', id: 'open_source_prizes' },
        { type: 'doc', id: 'private_prizes' },
        { type: 'doc', id: 'submissions_closed' },
      ],
    },
  ],
};

module.exports = sidebars;
