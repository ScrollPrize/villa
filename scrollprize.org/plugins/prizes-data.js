const path = require('path');
const fs = require('fs');
// js-yaml ships as a transitive dependency of Docusaurus (used by gray-matter
// et al.) — resolved from node_modules, no new package.json dependency.
const yaml = require('js-yaml');

/**
 * prizes-data — exposes the structured `prizes:` list maintained in the
 * frontmatter of docs/34_prizes.md as Docusaurus global data, consumed on the
 * landing page via `usePluginData('prizes-data')`.
 *
 * The prizes page body stays the single source of truth for humans; its
 * frontmatter carries the machine-readable mirror so the landing's open-prize
 * board updates automatically whenever the prizes page is edited.
 *
 * Consistent with this repo's plugin philosophy (see fetch-substack-posts.js,
 * atlas-data.js): the build FAILS LOUDLY if the frontmatter block is missing,
 * unparseable, or structurally invalid — never silently renders stale/empty
 * prize data.
 */

const SOURCE = 'docs/34_prizes.md';

function fail(message) {
  throw new Error(`[prizes-data] ${message}`);
}

function validatePrize(prize, index) {
  const where = `${SOURCE} frontmatter prizes[${index}]`;
  if (!prize || typeof prize !== 'object') {
    fail(`${where} is not an object`);
  }
  if (typeof prize.title !== 'string' || !prize.title.trim()) {
    fail(`${where}: "title" (non-empty string) is required`);
  }
  const label = `${where} ("${prize.title}")`;
  if (typeof prize.amount !== 'number' || !Number.isFinite(prize.amount) || prize.amount <= 0) {
    fail(`${label}: "amount" (positive number) is required`);
  }
  if (typeof prize.cadence !== 'string' || !prize.cadence.trim()) {
    fail(`${label}: "cadence" (non-empty string) is required`);
  }
  if (typeof prize.href !== 'string' || !prize.href.startsWith('/')) {
    fail(`${label}: "href" (site-relative path starting with "/") is required`);
  }
  if (typeof prize.hook !== 'string' || !prize.hook.trim()) {
    fail(`${label}: "hook" (non-empty string) is required`);
  }
  if (prize.unit !== undefined && (typeof prize.unit !== 'string' || !prize.unit.trim())) {
    fail(`${label}: optional "unit" must be a non-empty string`);
  }
  if (prize.featured !== undefined && typeof prize.featured !== 'boolean') {
    fail(`${label}: optional "featured" must be a boolean`);
  }
  if (prize.tiers !== undefined) {
    if (!Array.isArray(prize.tiers) || prize.tiers.length === 0) {
      fail(`${label}: optional "tiers" must be a non-empty list`);
    }
    prize.tiers.forEach((tier, tierIndex) => {
      if (
        !tier ||
        typeof tier.name !== 'string' ||
        !tier.name.trim() ||
        typeof tier.amount !== 'number' ||
        !Number.isFinite(tier.amount) ||
        tier.amount <= 0
      ) {
        fail(`${label}: tiers[${tierIndex}] needs "name" (string) and "amount" (positive number)`);
      }
    });
  }
}

module.exports = function prizesDataPlugin(context) {
  return {
    name: 'prizes-data',

    async loadContent() {
      const filePath = path.join(context.siteDir, SOURCE);
      if (!fs.existsSync(filePath)) {
        fail(`${filePath} not found`);
      }
      const raw = fs.readFileSync(filePath, 'utf8');

      // Frontmatter = the YAML between the first pair of `---` fences.
      const match = raw.match(/^---\r?\n([\s\S]*?)\r?\n---/);
      if (!match) {
        fail(`${SOURCE} has no frontmatter block (expected leading "---" fences)`);
      }

      let frontmatter;
      try {
        frontmatter = yaml.load(match[1]);
      } catch (err) {
        fail(`could not parse ${SOURCE} frontmatter YAML: ${err.message}`);
      }

      const prizes = frontmatter && frontmatter.prizes;
      if (!Array.isArray(prizes) || prizes.length === 0) {
        fail(
          `${SOURCE} frontmatter is missing a non-empty "prizes:" list — ` +
            'the landing page open-prize board is sourced from it',
        );
      }
      prizes.forEach(validatePrize);

      return { prizes };
    },

    async contentLoaded({ content, actions }) {
      actions.setGlobalData({
        prizes: content.prizes,
        generatedFrom: SOURCE,
      });
      console.log(
        `✓ [prizes-data] exposed ${content.prizes.length} open prizes from ${SOURCE}`,
      );
    },
  };
};
