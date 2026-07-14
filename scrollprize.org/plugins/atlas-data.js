const path = require('path');
const fs = require('fs');

/**
 * atlas-data — emits one static, SSR-rendered detail route per scroll
 * (`/data_browser/<id>`) from the generated `static/data_browser/index.json`.
 *
 * The index grid itself fetches that same JSON client-side (see
 * src/components/atlas/AtlasBrowser.js); this plugin only owns the per-scroll
 * pages so each gets a real URL with its own <head>/JSON-LD (good for SEO and
 * social unfurls). The JSON is produced by scripts/genAtlasData.js, which runs
 * before docusaurus in the `start`/`build` npm scripts.
 */
module.exports = function atlasDataPlugin(context, options) {
  return {
    name: 'atlas-data',

    async loadContent() {
      const dataPath = path.join(__dirname, '../static/data_browser/index.json');
      if (!fs.existsSync(dataPath)) {
        throw new Error(
          `[atlas-data] ${dataPath} not found. Run "yarn gen:atlas" (or "node scripts/genAtlasData.js") first.`,
        );
      }
      const data = JSON.parse(fs.readFileSync(dataPath, 'utf8'));
      if (!Array.isArray(data.scrolls)) {
        throw new Error('[atlas-data] index.json is missing a "scrolls" array');
      }
      return data;
    },

    async contentLoaded({ content, actions }) {
      const { createData, addRoute, setGlobalData } = actions;
      const { scrolls = [], _general = '' } = content || {};

      // The set of scroll ids that got a static detail route this build. The
      // index grid (AtlasBrowser) uses it to hide any live-only scroll that
      // appears in the fresher metadata.min.json but has no page here yet, so a
      // card never links to a 404. New scrolls surface at the next build.
      setGlobalData({ routedIds: scrolls.map((s) => s.id) });

      await Promise.all(
        scrolls.map(async (scroll) => {
          // Each detail route gets its own code-split JSON module (props.scroll).
          const dataPath = await createData(
            `atlas-scroll-${scroll.id}.json`,
            JSON.stringify({ ...scroll, _general }),
          );
          addRoute({
            path: `/data_browser/${scroll.id}`,
            component: '@site/src/components/atlas/ScrollDetailPage.js',
            exact: true,
            modules: { scroll: dataPath },
            // Feeds sitemap lastmod off the data_browser source page.
            metadata: { sourceFilePath: 'docs/02_data_browser.mdx' },
          });
        }),
      );

      console.log(`✓ [atlas-data] emitted ${scrolls.length} scroll detail routes`);
    },
  };
};
