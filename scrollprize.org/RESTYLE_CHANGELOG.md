# Website Restyle Changelog — "Obsidian Minimal" (July 2026)

Branch: `website-restyle`. Every commit on this branch is logged here: what changed, why, files touched, verification.
Direction: neutral true-dark competition-platform aesthetic (#0E0F11 bg, single #E5502B ember accent, hairline borders, flat surfaces, system fonts, tight type scale). Hard constraints: **no URL ever dies** (press links) and **seamless mobile** (360–768px verified).

Pipeline: multi-agent (design audit from full-page screenshots → token foundation → parallel work packages → QA gates: build, route crawl, URL preservation, console, page weight, bundle size, Tailwind purge tripwire, mobile compatibility, visual sign-off).

---

## C0 — `daeadb1b7` fix broken empty poster ref (404)

**Why:** the only console error on the baseline site: `docs/26_grandprize.md` referenced `poster="/img/grandprize/.webp"` (empty basename) — a 404 on every `/grandprize` load, 20 errors across QA viewports. No poster file ever existed; the video autoplays.
**Files:** `docs/26_grandprize.md` (1 line).
**Verification:** baseline capture identified it; the fix removes the attribute; post-fix expectation is zero console errors site-wide.

## C1 — tokens: design-token foundation, Tailwind bridge, content-glob fix

**Why:** establish one source of truth for the new visual system before any component work; fix a latent Tailwind purge bug.
**What:**
- New `src/css/tokens.css` — all `--vc-*` design tokens (Obsidian palette, type, radii, spacing). New `src/css/utilities.css` — shared recipes (`.vc-card`, `.vc-cta`, `.vc-btn`, `.vc-stat-strip`, `.vc-admonition`, `.vc-kicker`, `.vc-media`, …). New `chrome.css`/`landing.css` stubs for the parallel work packages; CSS load order: tokens → utilities → custom → chrome → landing → imageZoom.
- `tailwind.config.js`: content globs now cover `.mdx`/`.jsx` (previously `.mdx`-only classes survived purge by coincidence — a hard prerequisite for the upcoming stale-doc removal); color bridge `bg/surface/raised/line/dim/faint/accent/gold` → `var(--vc-*)`.
- `src/css/custom.css`: every raw color re-pointed to tokens (this re-skins the whole site to #0E0F11/#E5502B); global heading reset 900/-0.05em → 700/-0.01em (h1–h2) and 650/0 (h3+); docs heading sizes per new type scale; sidebar width 230→260px; Atlas block untouched except `--card`/`--line` indirection (its gold/status colors are frozen data-vis).
**Files:** `tailwind.config.js`, `docusaurus.config.js` (customCss array only), `src/css/{tokens,utilities,chrome,landing,custom}.css`, `RESTYLE_CHANGELOG.md`.
**Verification:** `yarn build` exit 0; old `#1C1A1D` gone from src CSS; `.mdx` purge tripwire class present in emitted bundle; `--vc-*` tokens compiled.
**Notable mappings:** Infima 7-step red ramp collapsed to the single ember accent; code-line highlight now a raised surface instead of a dark tint; `firstletters` black bg → `--vc-bg`.

## C4 — nav + IA: real navbar, slim sidebar, TOC restoration, orphan pruning with redirects

**Why:** all navigation was crammed into one 35-link sidebar (external socials included) behind a hidden-navbar hack; ~⅓ of entries were closed-prize history advertising stale deadlines; long pages had no table of contents; three orphaned/duplicate docs added clutter.
**What:**
- **Real navbar** (was `items: []` + brand faked inside the sidebar): Prizes · Data · Tutorials · Milestones · Community ▾ (Discord/𝕏/Substack/Donate/Jobs) · right-aligned "Get Started" CTA (`vc-navbar-cta`). Brand-in-sidebar HTML hack deleted; desktop navbar-height-0 hack removed from custom.css (navbar height now 56px); leftover brand 50%-opacity dimming removed.
- **Sidebar** 15 → 8 top-level entries, zero external links; Milestones/Scrolls/Archive collapsed (`sidebarCollapsible` flipped to true); `ink_detection` + `grand_prize` moved into Archive (both closed 2023 prizes). Stale nav labels cleaned via `sidebar_label` (e.g. "$700k/$100k/$50k Grand Prize (Dec 31)" → "2023 Grand Prize (closed)").
- **TOC restored** on 16 long-form docs (`hide_table_of_contents` removed): master_plan, faq, all tutorials, winners, prizes, segmentation, livestream, grandprize, firstscroll, firstletters, community_projects. FAQ's hand-rolled `<TOCInline>` red-link index removed (the real TOC replaces it).
- **Orphans deleted WITH redirects** (no URL dies): `06_tutorial_thaumato.md` + `29_tutorial4.md` (duplicate "Segmentation - a different approach") → `/segmentation`; `open_problem_rep.md` → `/unwrapping`. 3 inbound links that pointed at `tutorial4` while talking about ink detection now point at the actual Ink Detection tutorial (`tutorial5`) — a pre-existing mislink, fixed.
- **Footer**: 3-column sitemap → one dense band (© + CC BY-NC · Discord/GitHub/Substack/𝕏/Jobs). GitHub link → `github.com/ScrollPrize`.
**Files:** `docusaurus.config.js`, `sidebars.js`, `src/css/custom.css` (navbar hack removal only), `docs/*` (frontmatter/labels, FAQ index, 2 link fixes, 3 deletions).
**Verification:** `yarn build` green under strict `onBrokenLinks: throw`; all 7 redirect routes generate; Discord URL verified against repo (`discord.gg/V4fJhvtaQn`).
**Known follow-ups:** `static/llms.txt` still references `/open_problem_rep` (URL stays alive via redirect; text updated in C5); grandprize page still carries its inline gradient h1 (P2 polish, iteration round).

## C3 — components + chrome: one card recipe, docs-page skin, tap targets

**Why:** five shared components each carried their own visual system (6-layer shadows, rounded-2xl, three reds per card); docs pages had no coherent chrome (115-char lines, red link walls, six admonition hues, sub-44px tap targets).
**What:**
- `LatestPosts.js`: 4 shadowed cards → dense "Updates" strip (title 15px/600 + date, tabular numerals; 4-up → 2-up → single column responsive). Fetch/fallback logic untouched.
- `TopCard.js` → flat `.vc-card` (now unused but kept); `PrizeCard.js` → flat card, $ amount is the single red; `BeforeAfter.js` → 2px ember divider + 24px round handle (drag/touch logic byte-identical); `TutorialsTop.js` → hairline-framed pipeline thumbs, ember active state (autoplay behavior kept).
- `chrome.css` (15 sections): 56px navbar skin (blur, hairline, filled `vc-navbar-cta` button, 44px hamburger, safe-area), sidebar skin (uppercase faint category labels, ember active rule, no filled blocks), TOC skin, body-link discipline (neutral text + ember-tinted underline — un-reds FAQ/data/master_plan), ONE admonition recipe (covers `<Admonition>` and `<details>`), table skin (surface header, row hairlines, overflow-x scroll), code blocks 13.5px mono on #131518, framed content images (lightbox verified unaffected), dense footer band with 44px targets, visible heading anchors (incl. touch), focus-visible ember rings, docs content measure 720px.
**Files:** `src/components/{LatestPosts,TopCard,PrizeCard,BeforeAfter,TutorialsTop}.js`, `src/css/chrome.css`.
**Verification:** full `yarn build` exit 0 on the combined tree (C4 + C3 + WP1-in-flight).
**Routed follow-ups:** `docs/15_winners.md` has ~15 hand-rolled `bg-[#444]` prize-card clones → convert in integration pass; optional prism theme swap dracula→vsDark (config) left for the iteration round.

## C2 — landing: rewrite to Obsidian, slim-down, decorative machinery deleted

**Why:** the landing was a 2,110-line, ~10,800px-tall, 15.2MB movie-splash page: fullscreen autoplay volcano video with gradient text over live lava, six copies of a radial-gradient text treatment, 6-layer shadows, scroll-linked fixed background layers, three sponsor-tier accent hues, and three screens of poster-card walls.
**What:**
- `Landing.js` 2,110 → 866 lines; data arrays extracted to new `landingData.js`. Deleted: StoryBackground fixed layers + scroll-opacity JS, all gradient-text copies, all multi-layer shadows, mix-blend-exclusion, tier colors, hover-translates, sepia filters, the 40s auto-pan (panorama now a static, manually scrollable strip).
- Hero: navbar-aware `min(calc(100vh - navbar), 860px)`; video demoted to 0.35-opacity backdrop (0.25 mobile) behind solid→transparent media scrims, poster-first, reduced-motion → poster only; plain h1 + one ember line; stat strip `$1,800,500 · 5 scrolls · 2 read`; filled Get Started + text CTA; Breaking card in the hero fold.
- Story: all 6 beats kept at reading scale (mono ember kickers, 2rem titles, 96px gaps, one 720px framed media per chapter, 1px timeline rule). Open prizes merged into the 2026 chapter. Awarded prizes: 6 poster cards → one flush card with dense rows (title / 20px avatar stack / tabular $). Created By → one credit line. Sponsors: every donor kept, neutral tier headings, 24px-avatar rows, Citizens dense 2-col. Team kept (scale fix). Partners → one grayscale row.
- Sidebar hidden on landing (`landing.css`); est. height ~10,800 → ~7,100–7,400px (−33%; remaining levers noted for iteration).
- Pre-existing bugs fixed: `prizes#…` anchors missing leading `/`, Sergei Pnev URL missing protocol, Neil Parikh avatar `.webp`→`.jpg` (file was jpg), hero video src missing leading `/`, real alt text on partner logos.
**Files:** `src/components/Landing.js`, `src/components/landingData.js` (new), `src/css/landing.css`.
**Verification:** babel/postcss parse clean; all 85 image refs exist; inbound anchors (`#sponsors`, `#educelab-funders`, …) preserved; full `yarn build` green (below).

## C5 — integration: dead-rule sweep, winners cards, dead media, gradient-text zero

**Why:** converge the parallel work packages — remove now-dead legacy CSS, bring the last off-recipe surfaces onto the system, and drop confirmed-dead media.
**What:**
- `custom.css` 509 → 475 lines: old `.hero` bg-image rule, orphaned `.pan-horizontal`/`@keyframes pan`, grandprize "PT Sans" font override, two dead `max-width !important` rules removed (each verified covered by chrome/landing.css first). Atlas block untouched.
- `docs/15_winners.md`: **174** hand-rolled `bg-[#444]` prize-card anchors converted to the `.vc-card` recipe with the single-red amount treatment (all titles/winners/amounts/links/images preserved).
- Gradient-text count is now **zero site-wide**: grandprize h1 (WP4) + firstletters/submissions_closed/firstscroll h1 spans (orchestrator) → plain `text-accent` emphasis; poster classes (`font-black tracking-tighter`) dropped; firstscroll image-overlay hex arbitraries → `border-line`/`bg-black/65`/`text-white/90`.
- TOC restored on 4 more docs (data, data_fragments, background, unwrapping); `static/llms.txt` updated to the live `/unwrapping` target.
- **Dead media deleted (~60MB)** after zero-reference re-verification: `static/vid/graph_solver.mp4` (44MB), `static/img/tutorials/meshlab-segment4.webm` (16MB). Both were unreachable from any page — no press link can exist.
- Prism dark theme `dracula` → `vsDark` (closer to the Obsidian palette).
**Files:** `src/css/custom.css`, `src/css/chrome.css` (comment), `docs/{15_winners,26_grandprize,22_firstletters,25_submissions_closed,36_firstscroll,02_data,02_data_fragments,10_background}.md`, `docs/32_unwrapping.mdx`, `static/llms.txt`, `docusaurus.config.js`, 2 static deletions.
**Verification:** `yarn build` green; residual gradient/shadow/hex-arbitrary sweep clean outside the frozen atlas.

## C5b — media: 17 heavy assets re-encoded, −106MB on referenced pages

**Why:** the site cannot feel lean at 15.2MB landing weight; the worst in-use assets were multi-MB GIFs and uncompressed PNGs (a 17MB image link target, a 12.9MB reaction GIF).
**What:** 17 conversions, **129.8MB → 23.5MB (−81.9%)**, every output ≤3MB (Gate 4 cap). Highlights: hero backdrop webm 2.81MB → 0.17MB (960px CRF40 — displayed at 0.35 opacity); `youssef_text_wbb` 17.5MB PNG → 2.57MB high-quality webp (text readability preserved); `luke-reaction.gif` 12.9MB → 2.0MB animated webp; 9 progress-report GIF/webm files → compact VP9. Markup semantics preserved: markdown images stay images (animated webp); `<img>`→`<video>` swaps only where sibling media on the same page already used that pattern.
**Rule honored:** all 17 originals stay on disk — externally hotlinked asset URLs keep working; only page references changed (11 files).
**Files:** `src/components/Landing.js`, `docs/{02_data_segments,04_tutorial1,15_winners,22_firstletters,26_grandprize}.md`, `docs/32_unwrapping.mdx`, 17 new files under `static/`.
**Verification:** ffprobe duration/dimension checks + frame-diff spot checks per asset; zero stale references (grep); `yarn build` green.
**Skipped/flagged:** `desktop-scan.gif` (7.5MB) is unreferenced — left untouched; next-heaviest in-use asset is `tutorials/second_seg_run.webm` (7.5MB) + four 3–4MB progress files — follow-up candidates.

## Baseline (pre-restyle, recorded 2026-07-03)

- `yarn build` green; 82 sitemap routes all HTTP 200.
- Landing (desktop, full scroll-flush): **15,248,955 bytes / 137 requests**.
- `build/` 624,524 KB; `build/assets` 22,824 KB.
- Console errors: 20 (all the C0 404). Mobile: no horizontal overflow; ~10 sub-44px tap targets per page (hamburger 30×30, footer links 29–35×32).
- 72 arbitrary-value Tailwind classes inventoried across docs (purge tripwire baseline); 1 pre-existing purge victim (`w-[70%]`, mdx-only) recorded.
