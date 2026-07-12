/**
 * Creating a sidebar enables you to:
 - create an ordered group of docs
 - render a sidebar for each doc of that group
 - provide next/previous navigation

 The sidebars can be generated from the filesystem, or explicitly defined here.

 Create as many sidebars as you want.
 */

// @ts-check

/** @type {import('@docusaurus/plugin-content-docs').SidebarsConfig} */
const sidebars = {
  // Single, unified menu used across the whole site so every important page is
  // reachable from one coherent navigation surface. External links (Discord,
  // Substack, X, Donate) live in the navbar Community dropdown + footer, not
  // here. Requires `sidebarCollapsible: true` in the docs preset so the
  // `collapsed: true` categories below actually start collapsed.
  overviewSidebar: [
    {
      type: 'category',
      label: 'Overview',
      collapsible: true,
      collapsed: false,
      link: { type: 'doc', id: 'landing' },
      items: [
        { type: 'doc', id: 'get_started' },
        { type: 'doc', id: 'prizes' },
        {
          type: 'category',
          label: 'Open Problems',
          collapsible: true,
          collapsed: false,
          items: [
            {
              type: 'doc',
              id: '2026_open_problems',
              label: 'Problems in-depth',
            },
          ],
        },
        {
          type: 'category',
          label: 'Tutorials',
          collapsible: true,
          collapsed: false,
          items: [
            { type: 'doc', id: 'tutorial_VC3D' },
            { type: 'doc', id: 'tutorial_spiral' },
            { type: 'doc', id: 'tutorial5' },
          ],
        },
      ],
    },
    {
      type: 'category',
      label: 'Data',
      collapsible: true,
      collapsed: false,
      link: { type: 'doc', id: 'data_browser' },
      items: [
        { type: 'doc', id: 'data' },
        { type: 'doc', id: 'data_datasets' },
      ],
    },
    { type: 'doc', id: 'faq' },
    {
      type: 'category',
      label: 'Milestones & Results',
      collapsible: true,
      collapsed: true,
      link: { type: 'doc', id: 'winners' },
      items: [
        { type: 'doc', id: 'firstscroll' },
        { type: 'doc', id: 'grandprize' },
        { type: 'doc', id: 'firstletters' },
        { type: 'doc', id: 'community_projects' },
      ],
    },
    {
      type: 'category',
      label: 'Extra',
      collapsible: true,
      collapsed: true,
      items: [
        { type: 'doc', id: 'villa_model' },
        { type: 'doc', id: 'livestream' },
      ],
    },
    {
      type: 'category',
      label: 'Archive',
      collapsible: true,
      collapsed: true,
      items: [
        { type: 'doc', id: 'master_plan' },
        { type: 'doc', id: 'tutorial' },
        { type: 'doc', id: 'tutorial1' },
        { type: 'doc', id: 'tutorial2' },
        { type: 'doc', id: 'tutorial3' },
        { type: 'doc', id: 'segmentation' },
        { type: 'doc', id: 'unwrapping' },
        { type: 'doc', id: 'tutorial_VC' },
        { type: 'doc', id: 'grand_prize' },
        { type: 'doc', id: 'ink_detection' },
        { type: 'doc', id: 'open_source_prizes' },
        { type: 'doc', id: 'private_prizes' },
        { type: 'doc', id: '28_2024_prizes' },
        { type: 'doc', id: '30_2024_gp_submissions' },
        { type: 'doc', id: 'submissions_closed' },
      ],
    },
  ],
};

module.exports = sidebars;
