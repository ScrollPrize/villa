const path = require('path');
const fs = require('fs');

/**
 * winners-data — exposes the total dollars awarded, computed by summing every
 * money heading (`### \$60,000 …` / `## \$…`) on docs/15_winners.md, as
 * Docusaurus global data (`usePluginData('winners-data')`).
 *
 * The winners page stays the single source of truth: adding a new awarded
 * section with a `\$…` heading updates every "awarded" figure site-wide
 * (landing hero stat, teaser strip, tagline, meta descriptions, FAQ, and the
 * winners page's own headline) on the next build.
 *
 * Consistent with this repo's plugin philosophy (prizes-data.js et al.): the
 * build FAILS LOUDLY if the page can't be parsed or the total looks wrong —
 * never silently renders a stale/empty figure.
 */

const SOURCE = 'docs/15_winners.md';
// Sanity floor: the total can only grow. If parsing ever yields less than the
// July 2026 figure, the page structure changed and the regex needs attention.
const MIN_EXPECTED_TOTAL = 1_800_500;
const MIN_EXPECTED_SECTIONS = 20;

function fail(message) {
  throw new Error(`[winners-data] ${message}`);
}

function computeAwardedTotal(siteDir) {
  const filePath = path.join(siteDir || path.join(__dirname, '..'), SOURCE);
  if (!fs.existsSync(filePath)) {
    fail(`${filePath} not found`);
  }
  const raw = fs.readFileSync(filePath, 'utf8');
  // Money headings: "## \$12,000 …" / "### \$60,000 …" (the $ is KaTeX-escaped).
  const amounts = [...raw.matchAll(/^#{2,3}\s+\\\$([0-9][0-9,]*)/gm)].map((m) =>
    parseInt(m[1].replace(/,/g, ''), 10),
  );
  if (amounts.length < MIN_EXPECTED_SECTIONS) {
    fail(
      `only ${amounts.length} money headings found in ${SOURCE} ` +
        `(expected ≥ ${MIN_EXPECTED_SECTIONS}) — did the heading format change?`,
    );
  }
  const total = amounts.reduce((sum, n) => sum + n, 0);
  if (total < MIN_EXPECTED_TOTAL) {
    fail(
      `computed awarded total $${total.toLocaleString('en-US')} is below the ` +
        `known floor $${MIN_EXPECTED_TOTAL.toLocaleString('en-US')} — did the heading format change?`,
    );
  }
  return { total, sections: amounts.length };
}

module.exports = function winnersDataPlugin(context) {
  return {
    name: 'winners-data',

    async loadContent() {
      return computeAwardedTotal(context.siteDir);
    },

    async contentLoaded({ content, actions }) {
      actions.setGlobalData({
        awardedTotal: content.total,
        generatedFrom: SOURCE,
      });
      console.log(
        `✓ [winners-data] awarded total $${content.total.toLocaleString('en-US')} ` +
          `from ${content.sections} sections of ${SOURCE}`,
      );
    },
  };
};

module.exports.computeAwardedTotal = computeAwardedTotal;
