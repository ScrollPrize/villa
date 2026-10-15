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
  // One sidebar per navbar section; the sidebar is a "within this section"
  // nav, not a second copy of the site menu. Docs that aren't listed in any
  // sidebar (landing, get_started, prizes, open problems, faq, press, jobs,
  // villa_model, livestream) render standalone with no left sidebar.
  // Retired pages stay reachable inside the section they belong to, in a
  // collapsed archive group, because journals and Discord still link them.
  // A navbar item opens a section via `type: 'docSidebar'` (or, for the
  // Tutorials dropdown, src/theme/NavbarItem/DropdownNavbarItem), which also
  // keeps the item highlighted on every page of its sidebar.
  tutorialsSidebar: [
    {
      type: 'category',
      label: 'Tutorials & In-depth',
      collapsible: false,
      items: [
        { type: 'doc', id: 'tutorial_VC3D' },
        { type: 'doc', id: 'tutorial_spiral' },
        { type: 'doc', id: 'winding_annotations' },
        { type: 'doc', id: 'tutorial5' },
        {
          type: 'link',
          label: 'Scroll Anatomy',
          href: 'https://github.com/ScrollPrize/villa/blob/main/SCROLL_ANATOMY.md',
        },
      ],
    },
    {
      type: 'category',
      label: 'Older tutorials',
      collapsible: true,
      collapsed: true,
      items: [
        { type: 'doc', id: 'tutorial' },
        { type: 'doc', id: 'tutorial1' },
        { type: 'doc', id: 'tutorial2' },
        { type: 'doc', id: 'tutorial3' },
        { type: 'doc', id: 'segmentation' },
        { type: 'doc', id: 'unwrapping' },
        { type: 'doc', id: 'tutorial_VC' },
      ],
    },
  ],
  dataSidebar: [
    {
      type: 'category',
      label: 'Data',
      collapsible: false,
      items: [
        { type: 'doc', id: 'data_browser' },
        { type: 'doc', id: 'data' },
        { type: 'doc', id: 'data_datasets' },
      ],
    },
  ],
  milestonesSidebar: [
    {
      type: 'category',
      label: 'Milestones & Results',
      collapsible: false,
      items: [
        { type: 'doc', id: 'winners', label: 'Prize Winners' },
        { type: 'doc', id: 'firstscroll' },
        { type: 'doc', id: 'grandprize' },
        { type: 'doc', id: 'firstletters' },
        { type: 'doc', id: 'community_projects' },
      ],
    },
    {
      type: 'category',
      label: 'Past prize rounds',
      collapsible: true,
      collapsed: true,
      items: [
        { type: 'doc', id: '30_2024_gp_submissions' },
        { type: 'doc', id: '28_2024_prizes' },
        { type: 'doc', id: 'submissions_closed' },
        { type: 'doc', id: 'grand_prize' },
        { type: 'doc', id: 'open_source_prizes' },
        { type: 'doc', id: 'private_prizes' },
        { type: 'doc', id: 'ink_detection' },
        { type: 'doc', id: 'master_plan' },
      ],
    },
  ],
};

module.exports = sidebars;
