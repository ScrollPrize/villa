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

## C6 — qa fixes: mobile drawer, footer band, landing height, polish

**Why:** the visual sign-off found one blocker (empty mobile drawer) plus prioritized polish; Gate 3 traced 10 console 404s to a pre-existing broken poster ref.
**What:**
- **Mobile drawer fixed** (the blocker): the navbar's `backdrop-filter` established a containing block that trapped the fixed-position drawer at 56px height, clipping all content. Fixed with `100dvh` viewport-relative heights; drawer panels get solid `--vc-bg` + hairline; the "Back to main menu" affordance (hidden by legacy CSS, stranding users in the docs panel) restored as a 44px row. Hamburger no longer forced visible on desktop.
- **Footer** as the spec's dense band: one row ≥768px (© left, links right), 44px-wide tap targets on narrow labels, navbar brand hit-area ≥44px. Verified: **0 small tap targets** (baseline ~10/page, after-1 3/page).
- **Landing height**: values pass (media 200px, tighter credits rhythm, hero cap) + folding of bulk lists behind expanders with counts — Citizens sponsors (28), Advisors & Alumni (17), Papyrology Advisors (6), and the three pre-2023 story beats under a "The backstory" expander (all content preserved, expandable in place). Desktop height now **7,270px** (baseline ~10,800, −33%).
- `/tutorial` console 404 fixed: `TutorialsTop.js` referenced a poster (`top-prediction-small3.webp`) that never existed — now points at the real file. Pre-existing bug (present on main).
- Polish: `<details>` skin hairline (ember reserve), date suffixes dropped from milestone sidebar labels, `/prizes` blockquote → standard `:::warning` admonition, landing heading anchors hidden. Firstscroll page content refresh (hero video webm+poster, og image) included.
**Files:** `src/css/{chrome,landing}.css`, `src/components/{Landing,TutorialsTop}.js`, `docs/{22_firstletters,26_grandprize,34_prizes,36_firstscroll}.md`.
**Verification:** build green; targeted capture: drawer populated on solid bg (screenshot-verified), no desktop hamburger, 0 overflow / 0 small taps / 0 console errors on `/` and `/get_started` × 5 viewports; landing single-tile (<8,000px).

## C7 — owner requests: real stats, PHerc 1667, deeper IA pruning, landing folds

**Why:** direct instructions from Giorgio during the iteration round (given to the fix agent mid-run).
**What (all seven requests):**
1. Hero stat strip now uses real counts — `$1,800,500 awarded · 35 scrolls scanned · 1 scroll fully read` (35 sourced from the atlas dashboard totals); stale copy fixed; 2026 story chapter retitled "The first scroll is read." featuring PHerc. 1667 → `/firstscroll`.
2. Landing folds (all content preserved behind native `<details>` expanders with counts): Citizens sponsors (28), Advisors & Alumni (17), Papyrology Advisors (6), and the 79 AD–2015 "backstory" story beats. Final landing height **6,971px desktop** (baseline ~10,800, −35%) / 10,265px mobile (from 14,226).
3. Segments + Fragments data pages deleted; redirects `/data_segments→/data`, `/data_fragments→/data`; 6 inbound refs fixed; Data sidebar slimmed.
4. Curated Datasets page gains the standard date line (July 2025) under its h1.
5. History page (`/background`) deleted with redirect → `/` (the landing's "Our story" covers it); "The Scrolls" sidebar category kept holding Villa model + Livestreams; FAQ/grandprize refs repointed.
6. Jobs page gains a "start as a contributor" note (open prizes + Discord path).
7. EduceLab funders block removed from the landing (partners row stays; firstletters anchor repointed to `/#partners`).
**Files:** `src/components/Landing.js`, `src/css/{landing,chrome}.css`, `sidebars.js`, `docusaurus.config.js`, 11 docs edited, 3 docs deleted.
**Verification:** build green (strict link check); redirect stubs serve 200 for all three removed slugs (no URL dies); harness on /, /data, /faq, /jobs + /get_started: 30/30 combos with 0 console errors, 0 overflow, 0 small tap targets; 773 local media refs across 87 pages all resolve.

## C8 — owner requests round 2: hero CTAs, clickable stats, story trim, team layout

**Why:** direct instructions from Giorgio during local validation.
**What:**
1. "Join Discord" outline button added next to the filled Get Started in the hero CTA row.
2. The "35 scrolls scanned" and "1 scroll fully read" hero stats are now links opening the data browser (`/data_browser`); hover shows the ember accent.
3. Story: the "2024 AD — New frontiers" chapter deleted (its two images stay on disk); timeline is now backstory-expander → 2023 → 2026.
4. The 2026 chapter shows the unwrapped PHerc. 1667 banner (`banner-strip.webp` from the firstscroll post, 256KB, displayed as a 200px pannable strip linking to `/firstscroll`) and no longer mentions the Master Plan.
5. Team layout: "Vesuvius Challenge Team" and the renamed "Vesuvius Challenge Papyrology Team" sit side by side; "EduceLab Team" is now collapsible and sits beside "Advisors & Alumni"; Papyrology Advisors unchanged below. Dead `.vc-story__pair` CSS removed.
**Files:** `src/components/Landing.js`, `src/css/landing.css`.
**Verification:** build green; landing capture at 5 viewports: 0 overflow, 0 small tap targets, 0 console errors; desktop height now 6,447px; layout screenshot-verified.

## C9 — owner requests round 3: navbar Discord button, banner crop, Senators fold

1. "Join Discord" is now also a navbar button (outline style, right side, next to the filled Get Started — the one-filled-element rule holds; drawer clone styled to match; auto external-link icon hidden for the button look). `docusaurus.config.js` + `chrome.css`.
2. The 2026 PHerc. 1667 banner is a fixed crop of the middle-right region (`object-fit: cover; object-position: 70% 50%`) instead of a left-edge pannable strip — the sparse left edge looked poor; full banner remains one click away on `/firstscroll`. `Landing.js` + `landing.css`.
3. The "Senators" sponsor tier is collapsible like Citizens. `Landing.js`.
**Verification:** build green; landing capture ×5 viewports: 0 overflow / 0 small taps / 0 console errors; screenshot-verified.

## C10 — owner requests round 4: open problems restructure, Rocío art removal

1. Open problems section reduced to two clearly separate cards: **Virtual Unwrapping** (renamed from "Geometric Reconstruction"; same slider, links, skills) and **Ink Detection**. The "Representation" card removed (its `/unwrapping` destination remains reachable via the Virtual Unwrapping card; slider images stay on disk).
2. The Villa dei Papiri illustration by Rocío Espín removed from the 79 AD story beat, along with the artist-credit caption above Partners; dead `.vc-team__credit` CSS removed. (Asset stays on disk per hotlink policy.)
**Files:** `src/components/Landing.js`, `src/css/landing.css`.
**Verification:** build green; landing capture ×5 viewports clean (0 overflow / 0 small taps / 0 console errors); desktop height now 5,867px; screenshot-verified.

## C11 — owner requests round 5: open-problem captions grounded in dashboard data, "Extra" category

1. Both open-problem captions rewritten to be internally consistent and current, grounded in the data browser dashboard (updated 2026-06-15: 45 scanned = 35 scrolls + 10 fragments, 12 in segmentation, 9 with ink results, 4 with recovered text): **Virtual Unwrapping** no longer opens with a dangling reference to the removed Representation card, uses "unwrapped" terminology, cites PHerc. 1667 end-to-end and the 12-of-45 segmentation gap; **Ink Detection** now says ink surfaced on 9 of 45 and text recovered from 4 (was "four of the 35"), keeping the "if it ever existed, it can be detected" close.
2. Sidebar category "The Scrolls" renamed to **"Extra"** (still holds Villa Model + Livestreams).
**Files:** `src/components/Landing.js`, `sidebars.js`.
**Verification:** build green; captions confirmed in served HTML; capture ×2 routes ×5 viewports clean.

## C12 — owner requests round 6: sidebar default-expansion policy

Sidebar now opens with only **Overview** and **Data** expanded by default; Tutorials, Milestones & Results, Extra, and Archive start collapsed and auto-expand only when the visited page lives inside them (Docusaurus native active-path expansion). Change: `Data` category `collapsed: true → false` in `sidebars.js` (Overview was already expanded; the rest already collapsed).
**Verification:** screenshots — `/get_started` shows Overview+Data open, others closed; `/tutorial1` additionally shows Tutorials auto-expanded with the active item highlighted.

## C13 — fix: firstscroll banner distortion (chrome image rule vs pannable strip)

The docs-chrome rule framing content images (`max-width:100%` + border) was crushing the 4000px-wide PHerc. 1667 banner on `/firstscroll` into the 720px column while its height stayed pinned — visible distortion. Images marked `max-w-none` or living inside an `overflow-x-auto` strip are now exempt (natural width, panning restored, no doubled frame — the frame belongs to the outer container).
**Files:** `src/css/chrome.css`.
**Verification:** build green; `/firstscroll` screenshot shows the strip at natural proportions with scale-bar/credit overlays and sideways panning.

## C14 — owner request: archive the Master Plan

Master Plan moved from the Overview sidebar category into Archive (first entry). URL `/master_plan` unchanged — no redirect needed; inbound links unaffected (the landing already stopped referencing it in C8).
**Files:** `sidebars.js`.
**Verification:** sidebars.js loads clean via node; doc id resolves (full build rides with the in-flight prizes/sticker round, which the strict checker covers).

## C15 — prizes: 2027 Grand Prize + restructured prize suite + dynamic prizes plugin

**Why:** announce the 2027 Grand Prize and rebuild the whole prize offering around it (owner-directed, July 4–6 sessions).
**What:**
- **2027 Grand Prize** section (from the owner's spec document): \$800k/\$100k/\$50k/\$50k = \$1M pool, 23 eligible PHerc volumes (each linking to its Data Browser page + neuroglancer), June 25th 2027 deadline; "Eligible scroll volumes" and "Submission criteria and requirements" as collapsibles; updated rules (70%-per-column letter-by-letter legibility, concurrent-submission ranking, merged skip-patches clause, multi-scroll-dataset requirement dropped).
- **First Letters** restructured: \$50k per scroll across the GP volume set (\$40k first + \$10k second team per scroll), max 10 scrolls = up to \$500k; 2023-rubric features restored (purpose statement, "rather be slow than wrong", per-letter dimensions, 3D orientation, rows/fiber-overlay soft criteria, held-out validation, stay-open clause). Split into its own section.
- **PHerc. Paris 4's Title Prize** (renamed from First Title — titles were already found in PHerc. 172/139): \$50k, any scan including 2.4µm, framed with Data Browser facts; own section + criteria collapsible.
- **Progress Prizes**: \$20,000 guaranteed monthly best-submission award added (pool math now \$590k/yr).
- **NEW `plugins/prizes-data.js`**: reads the prizes list from `docs/34_prizes.md` frontmatter at build time and exposes it as global data — the landing's prize surfaces update automatically with the prizes page. Total open pool: **\$2,140,000**.
- Entity rename in prize T&C: Curious Cases, Inc. → Scroll Prize, Inc.
**Files:** `docs/34_prizes.md`, `plugins/prizes-data.js` (new), `docusaurus.config.js` (plugin registration).

## C16 — landing: prize-pool sticker, open-prizes board, hero stats, team overhaul

- **Floating sticker** (bottom-center pill on desktop, bottom bar on phones): dynamic **\$2,140,000 open prize pool**, clickable → /prizes; scroll-triggered entrance, yields while the Open Prizes board or footer is on screen, dismissible (localStorage), reduced-motion aware, `SHOW_PRIZE_STICKER` toggle.
- **Open prizes board** ("\$2,140,000 prize pool") in its own section directly above Open Problems, fed by the plugin; per-prize rows with ember amounts, hooks, tier chips, deadlines.
- **Hero stat strip**: dynamic open-pool stat leads; `$1,800,500 already awarded` → links to /winners; scroll stats → Data Browser. **Awarded Prizes section removed** (winners page replaces it).
- **Open problems**: owner-dictated concise scientific captions; prize tie-in lines; tags cleaned; targets strip tiles link to the villa monorepo.
- **Story**: 2023 chapter condensed to one paragraph.
- **Team**: two-column credits (Created by list with roles — Nat Friedman Instigator, Director & Founding Sponsor; Daniel Gross Founding Sponsor; Dr. Brent Seales Principal Advisor | Led by Giorgio Angelotti, Project & Tech Team Lead, PhD); masonry team columns; Tech Team (renamed) + new Annotation Team; Papyrology Team renamed + titles de-duplicated; EduceLab Team (Partners); Youssef Nader → Advisors & Alumni; Daniel Gross and duplicate Federica Nicolardi entries removed; Senators+Citizens sponsor expanders side-by-side.
**Files:** `src/components/Landing.js`, `src/components/landingData.js`, `src/css/landing.css`.

## C17 — atlas: human-readable PHerc labels

Scrolls display as "PHerc. 172" / "PHerc. Paris 4" instead of raw ids ("PHerc0172"): `label` derived at generation time (`phercLabel()` in `scripts/genAtlasData.js`), rendered in cards, detail pages, tab titles, JSON-LD, alt/aria texts; search matches both spellings. **URLs unchanged** (no-dead-links rule).
**Files:** `scripts/genAtlasData.js`, `src/components/atlas/{ScrollCard,ScrollDetailPage,AtlasBrowser}.js`.

## C18 — tutorials & IA: outdated banners, VC archive, pipeline strip fix

- **[OUTDATED CONTENT] admonition** on all tutorials (hub + Scanning/Representation/Seg-and-Flattening/Ink Detection/Segmentation/VC), Virtual Unwrapping, and Curated Datasets.
- **Volume Cartographer tutorial archived** (sidebar → Archive; URL unchanged); sidebar tutorial order matches the pipeline strip (Ink Detection last); tutorial3 highlights the correct strip step.
- **Pipeline strip renders in one row** (thumbs shrink instead of wrapping — 4×150px+arrows exceeded the 720px measure).
**Files:** `docs/{03_tutorial,04_tutorial1,05_tutorial2,06_tutorial3,06_tutorial_VC,07_tutorial5,35_segmentation}.md`, `docs/32_unwrapping.mdx`, `docs/02_data_datasets.md`, `sidebars.js`, `src/components/TutorialsTop.js`.

## C19 — get started & FAQ: brought current (July 2026)

- **Get Started**: 2026 PHerc. 1667 hook; "Dive in" leads with the 2027 Grand Prize; Ink Detection before Segmentation; Data Browser as the see-a-scroll destination; Discord invite unified (`discord.gg/V4fJhvtaQn`); `vesuvius` Python + VC3D recommended (C library mentions removed); \$20k/month noted in Open Source box.
- **FAQ**: OUTDATED banner on top; intro + dates answers rewritten around the 2026 milestone and 2027 deadlines; publications list adds the 2026 PHerc. 1667 preprint and **Angelotti et al., Scientific Reports 2026** (nature.com + arXiv); texts answer points at Data Browser readings (172 *On Vices*, 139 *On Gods*); software answer → VC3D + `vesuvius`; scan-parameter claims reframed as history (7.91µm-era vs 2.4µm/1.1µm today); slice-orientation section marked as tif-stack-era; Discord unified; Scroll Prize, Inc. rename.
**Files:** `docs/01_get_started.md`, `docs/09_faq.md`.

## C20 — First Letters flattened to \$50k first-team-only

The \$40k/\$10k first/second-team split is removed: **\$50,000 per scroll to the first team** (10 letters in 4 cm², open-source to accept). Max 10 scrolls / \$500k total unchanged; tier chips removed from the landing board row.
**Files:** `docs/34_prizes.md` (headline + frontmatter).

## C21 — open-prize-pool banner atop the Prizes page

New `PrizePoolBanner` component at the top of /prizes: ember-bordered card showing the plugin-computed total ("Open prize pool · \$2,140,000 · Grand Prize deadline June 25th, 2027 →"), linking to the 2027 Grand Prize section. Same data source as the landing sticker/board — one value everywhere.
**Files:** `src/components/PrizePoolBanner.js` (new), `docs/34_prizes.md` (import + placement).

## C22 — data browser: segment folder links + copy-S3 buttons (community feedback)

Per Sean (bruniss) and Forrest's feedback: the per-segment "mesh ↗" .obj link (no current tool consumes .obj, and the linked obj was wrong) is replaced by **"files ↗"** — the segment's root folder (containing the tifxyz, surface volumes, and predictions) in the S3 file browser — plus a **"copy s3"** button that puts the rclone-able `s3://…/segments/<id>/` path on the clipboard. The folder is derived at generation time from any of the segment's data URLs.
**Files:** `scripts/genAtlasData.js`, `src/components/atlas/InkSegmentsGallery.js`.

## C23 — technical blogpost: "From Ash to Text" (July 2026)

New page `/from_ash_to_text` converted from the team's Google Doc — a technical map of the whole scanning → unwrapping → ink-detection pipeline and where the community can help.

- **Full image quality preserved**: the 26 figures come from the docx export's original media (up to 2048px wide, 23MB total), not the ~624px versions the markdown export embeds; built assets verified byte-identical to the originals.
- Six "🙋 How you can help" callouts converted to `:::tip[How you can help]` admonitions; editorial placeholders stripped; all `$` KaTeX-escaped; og/twitter meta set.
- **Placements**: sidebar Overview, directly above the outdated Virtual Unwrapping page; Get Started 2027-Grand-Prize box ("Starting point: From Ash to Text — the state of the pipeline") and Segmentation box (replaces the Virtual Unwrapping link); landing Virtual Unwrapping open-problem card CTA now "Chart the Path" → `/from_ash_to_text` (was /segmentation).
- Verified: build green, zero horizontal overflow at 390px, no console errors, admonitions and tables render.
**Files:** `docs/37_from_ash_to_text.md` (new), `static/img/ash2text/` (26 new images), `sidebars.js`, `docs/01_get_started.md`, `src/components/Landing.js`.

## C24 — blogpost route renamed to /tech_blogpost

The C23 page moves from `/from_ash_to_text` to **`/tech_blogpost`** (doc file, sidebar id, Get Started links, landing CTA, og/twitter URLs). A client redirect keeps `/from_ash_to_text` alive (it existed on preview builds). Title and sidebar label ("From Ash to Text") unchanged.
**Files:** `docs/37_tech_blogpost.md` (renamed from `37_from_ash_to_text.md`), `sidebars.js`, `docs/01_get_started.md`, `src/components/Landing.js`, `docusaurus.config.js` (redirect).

## C25 — blogpost named "Technical Blogpost" + navbar entry

- Page title → **"Technical Blogpost: Why Reading Every Herculaneum Scroll Is Still a Challenge"**; sidebar label and Get Started link texts → "Technical Blogpost"; og/twitter titles updated. Route stays `/tech_blogpost`.
- **New top-navbar item "Technical Blogpost"** between Prizes and Data.
- Navbar links get `white-space: nowrap` + tighter item padding in the 997–1220px band so the two-word label doesn't wrap just above the hamburger cutoff (verified single-line at 997/1024/1150/1440, zero overflow).
**Files:** `docs/37_tech_blogpost.md`, `docs/01_get_started.md`, `docusaurus.config.js` (navbar item), `src/css/chrome.css`.

## C26 — blogpost: clickable cross-references

- All textual cross-references in `/tech_blogpost` now link to their targets: three "(see References)" → `#references-and-implementation-links`; "Sections 2–4" / "Sections 2 and 3" → the numbered section anchors; "the previous section" (trail metaphor) → Meshes; the five recap bullets' quoted section names → their sections. The recap's stale name "Why approximate supervision is a bottleneck" corrected to the real heading "Label quality: the main unwrapping bottleneck".
- **Content headings demoted one level** (`#`→`##`, `##`→`###`) — the doc keeps a single h1 (the title), the major sections finally get anchor ids (content-level h1s got none, which the strict `onBrokenAnchors` build caught), and the TOC now shows the proper section→subsection hierarchy.
- Verified: build green (anchor checker), click-test lands headings just below the fixed navbar.
**Files:** `docs/37_tech_blogpost.md`.

## C27 — blogpost: Greek transcription caption un-mangled

The 1.1 µm vs 2.4 µm ink-comparison caption rendered as "AT 1.1 ΜM: ]ΟΥ̓ΔῈ ΓᾺΡ…" — the Google-Docs export emitted the two-column caption as a one-row markdown table, whose only row becomes a `<th>` header, and the site's table styling uppercases headers (µ → Μ, Greek capitalized). Replaced with a flex two-column italic caption (line breaks of the transcription restored), aligned under the two image panels; stacks on mobile. The doc's two real tables (scan parameters, bottlenecks) keep their intended uppercase headers.
**Files:** `docs/37_tech_blogpost.md`.

## C28 — blogpost: 🙋 icon on the "How you can help" boxes

The six callouts showed the default tip lightbulb; the post's own convention is the 🙋 raising-hand ("look for the 🙋 callouts"). The `:::tip` blocks are now `<Admonition type="tip" icon="🙋" title="How you can help">` (import added). Icon renders monochrome like every admonition emoji site-wide (intentional chrome.css treatment).
**Files:** `docs/37_tech_blogpost.md`.

## C29 — merge main into website-restyle

Brings the branch current with main: the data page rewrite (dataset naming, deprecated samples endpoint, 2026 pre-print reference, EduceLab "scanned at DLS before 2025" scoping), the PHerc. Paris 4 publication notice + per-scroll license scopes (PR #1117), and the PHerc. 1667 DLS→ESRF scan-history correction (PR #1118). One conflict in `docs/02_data.md` (a "scroll-only or fragment-only views" sentence main had deleted — deletion kept). The atlas auto-merge preserved both sides: restyle's `phercLabel` display labels + segment folder/copy-S3 links, and main's `legalNotice`/`licenseScope`/curated-licenses pipeline.
**Files:** merge commit (docs/02_data.md resolved by hand; genAtlasData.js, ScrollDetailPage.js, DataCatalog.js, atlasOverlay.json auto-merged).

## C30 — blogpost is now "Open Problems" at /2026_open_problems; Virtual Unwrapping archived

- **Virtual Unwrapping archived**: moved from the Overview sidebar category into Archive (after Master Plan). URL `/unwrapping` unchanged — still resolves.
- **Blogpost renamed "Open Problems"** everywhere it's displayed: navbar item (between Prizes and Data), sidebar label, page title ("Open Problems: Why Reading Every Herculaneum Scroll Is Still a Challenge"), og/twitter titles, and the two Get Started starting-point links.
- **Route moved to `/2026_open_problems`**: file renamed to `docs/37_2026_open_problems.md` with an explicit frontmatter `id` (the default number-prefix parser won't strip `37_` when the rest starts with a digit — same reason `28_2024_prizes` keeps its full id). Redirects updated so `/tech_blogpost` and `/from_ash_to_text` both point directly at the new route (no chained hops). Landing "Chart the Path" CTA updated.
**Files:** `docs/37_2026_open_problems.md` (renamed from `37_tech_blogpost.md`), `sidebars.js`, `docusaurus.config.js` (navbar + redirects), `docs/01_get_started.md`, `src/components/Landing.js`.

## C31 — Open Problems: pipeline gif strip as visual TOC

The Tutorials pipeline strip (Scanning → Representation → Segmentation and Flattening → Ink Detection) now also sits in the Open Problems post's "The pipeline" section, directly under the three-step list. On this page the four thumbs deep-link to the post's own section anchors (§1 Scanning, §2 Unwrapping ×2, §3 Ink recovery) — a visual table of contents — via a new optional `links` prop on `TutorialsTop` (tutorial pages keep their default destinations). Verified: thumbs land sections below the fixed navbar; mobile wraps 3+1 with zero overflow.
**Files:** `src/components/TutorialsTop.js`, `docs/37_2026_open_problems.md`.

## C32 — Open Problems strip labels match the post's section names

The pipeline strip on /2026_open_problems now reads Scanning → Unwrapping → Flattening → Ink recovery (was Representation / Segmentation and Flattening / Ink Detection), via a `labels` override prop on `TutorialsTop` mirroring the `links` one. Tutorial pages keep their original labels.
**Files:** `src/components/TutorialsTop.js`, `docs/37_2026_open_problems.md`.

## C33 — feedback round: Get Started trim, data list, landing links, academic references

- **Get Started**: removed the "Curious how far the field has come? … Kaggle Surface Detection winners" line.
- **Open Problems "Where is the data"** is now a list: Data Browser, direct S3 bucket, curated datasets (ink and spiral) at huggingface.co/buckets/scrollprize/datasets, model checkpoints at huggingface.co/scrollprize.
- **Landing Open-problems cards**: titles now link to the matching Open Problems sections (Virtual Unwrapping → §2, Ink Detection → §3) via a `titleHref` prop on ChallengeBox.
- **Open Problems references overhauled** (agent-audited, every URL curl-verified 200): the flat link list is now structured (Publications / Methods cited / Competition / Code / Model checkpoints / Data / Project) with academic citations — Angelotti et al. arXiv:2606.29085 (canonical /abs/ replaces the /html/ render), Angelotti et al. Scientific Reports 2026, EduceLab-Scrolls, nnU-Net (Isensee et al. 2021), U-Net, 3D U-Net, ResNet3D, Kinetics-700, DINO, Adam, Ceres, SLIM, Paganin 2002, WebKnossos, OME-NGFF. Thirteen in-text terms now link to their canonical sources. Fixed: dead AutoInk link (404 — removed), malformed "sc ripts" link text.
**Files:** `docs/01_get_started.md`, `docs/37_2026_open_problems.md`, `src/components/Landing.js`.

## C34 — owner feedback round 2: captions, FAQ revision, site-wide fix sweep

- **Open Problems captions**: 10 missing captions inserted (owner-approved; image5/6 identified as the same PHerc. Paris 4 compressed region at DLS 7.91 µm suboptimal vs ESRF 2.4 µm optimized protocol; image7 corrected to "binarized output"). Existing caption fixes: image17/18 "left/right pair" → "first/second pair (z=120/z=200)"; image24's "1)" (which rendered as a numbered list) → "a)" inside the italics; stray object-replacement char after image1 removed; 7.81 µm typo → 7.91 µm. New caption style (chrome.css, scoped to the page): italic paragraph after an image renders centered, dimmed, 65ch measure, bound to its image.
- **FAQ revised per FAQ_AUDIT.md**: OUTDATED banner → "WORK IN PROGRESS (July 8, 2026)"; segmentation-before-ink answer now reflects shipped direct 3D ink segmentation (links to the Open Problems sections); stale `.volpkg` path → Data Browser; Academic papers leads with the Open Problems post; ImageJ answer reframed around VC3D/Neuroglancer/`vesuvius`; `.hdf` claim past-tensed with OME-Zarr as current; tif windowing framed historically; dead `ttps://` algotom link fixed.
- **Site-wide fixes (sweep findings)**: prizes monthly deadline June 30 → July 31, 2026 and "most-read scroll" framing corrected; old Discord invites → discord.gg/V4fJhvtaQn (livestream, submissions_closed, grandprize); closed/historical banners added to /ink_detection, /2024_gp_submissions, /firstletters; 2024_prizes banner past-tensed; KaTeX display equations scroll in-container (mobile); /unwrapping fixed-600px iframe made responsive (226px mobile overflow → 0); landing ink card "9 of 45 scanned scrolls" → "scrolls and fragments"; tutorial link text matches its target, deprecated C library dropped, invalid webm poster removed; tutorial2/segmentation/grand_prize/community_projects typos; vesuvius-c marked deprecated + VC3D added to community tools; EduceLab paper dated 2023 on /data; winners' automated-segmentation heading dated (December, 2024); villa 3D viewer min-height on phones; data browser: Paris 4 work "attributed to Philodemus (title not yet identified)", stale segments=20 now falls back to the live S3 count, "— keV" empty-energies fixed, fragment pages say "About this fragment".
**Files:** 24 files (docs/, src/components/, src/css/chrome.css, src/data/atlasOverlay.json).

## C35 — awarded total + sample counts go dynamic; fragments stat on the hero

- **New `plugins/winners-data.js`**: sums every money heading on docs/15_winners.md (29 sections = \$1,800,500 exactly) and exposes it as global data; fails the build loudly if the parse looks wrong (floor \$1,800,500, ≥20 sections). Adding a new winner section now updates every "awarded" figure site-wide.
- Consumers: landing hero stat, LatestPosts teaser ("\$1.8M+" via a new `<AwardedTotal compact />` component), the winners page's own headline, the FAQ figure, and docusaurus.config.js (tagline, default meta description, org JSON-LD) via the exported `computeAwardedTotal()`.
- **`plugins/atlas-data.js` now exposes live sample counts** (scrolls/fragments/samples from the build-time index). Hero stat strip: "10 fragments scanned" added before "35 scrolls scanned", and both counts are now dynamic instead of hardcoded.
**Files:** `plugins/winners-data.js` (new), `src/components/AwardedTotal.js` (new), `plugins/atlas-data.js`, `src/components/Landing.js`, `src/components/LatestPosts.js`, `docs/15_winners.md`, `docs/09_faq.md`, `docusaurus.config.js`.

## C36 — ink detection tutorial rewritten for the koine_machines pipeline

- **Full rewrite of docs/07_tutorial5.md** (was a 2023-era overview with an OUTDATED banner, pointing at Kaggle notebooks): now a hands-on tutorial for the `ink-detection/koine_machines` pipeline (villa `merge-ink-pipelines` branch), in the style of tutorial2. Conceptual intro (signal recovery, iterative labeling, safeguards, arXiv 2606.29085) followed by dataset → setup → 2.5D training → inference → native-3D training + inference → scaling up → iterative labeling.
- Runs end-to-end on **one segment** (`phercparis4/w00_20231016151002`, ~25 GB) from the `ink-labels` HF bucket (cross-linked to the C-series curated-datasets entry on /data_datasets); native-3D section documents `volume_source.txt` with the 2.4 µm open-data S3 volume (anonymous reads verified in the pipeline code).
- **New figures rendered from real bucket data**: labels overlay (red), supervision-mask overlay (green), and the trained model's prediction with training labels overlaid — 1.3% of pixels labeled, ~10% predicted as ink.
- Every command and config key verified against the branch code; training/inference runs reproduced on cluster GPUs before writing. Note: the tutorial depends on ink-detection fixes that must land on `merge-ink-pipelines` (tifxyz reader fallback, lazy cuCIM import + empty-validation checkpoint save, full_3d_single_wrap surface-mask reconstruction in inference).
**Files:** `docs/07_tutorial5.md`, `static/img/tutorials/ink-labels-overlay-w00.webp` (new), `static/img/tutorials/ink-supervision-overlay-w00.webp` (new), `static/img/tutorials/ink-prediction-w00.webp` (new).

## C37 — ink tutorial: spiral cross-link, figure refresh, wording cleanup

- **Reciprocal link to the new [spiral tutorial](docs/38_tutorial_spiral.md)** (added upstream in `7de6f7f45`): the native-3D inference section now notes its scroll-space ink-prediction volume is exactly what the spiral fit's `render_ink.py` renders each winding through, linking to `tutorial_spiral#rendering-ink`. (Spiral → ink links already existed; this closes the loop.)
- **New/replaced figures**: intro now shows the color / 1000nm-infrared / X-ray comparison of one detached fragment (Parsons, *Hard-Hearted Scrolls*, Fig 1.3), replacing the old `ink1-alpha`/`ink2-alpha` — makes the "ink is invisible in X-ray, but the signal is there" point concretely, plus a paragraph on why we scan with X-rays (penetration vs. contrast). "How ink detection works" gains a **crackle / ink-signal** figure (PHerc. 1667 paper Fig 2) with a link to `/firstletters` (Casey Handmer's crackle discovery). The iterative-labeling section ends with the **pseudo-labeling process** figure (Model 0→5, labels/prediction/held-out rows).
- **Wording cleanups**: dropped "so there is nothing to install system-wide"; moved the cuCIM/CuPy `:::info` from setup to the native-3D training section where dilation is relevant; simplified the `volume_path` local-download note (removed the "considerably faster" hand-holding).
- `ink1-alpha.webp` / `ink2-alpha.webp` are now orphaned (no references) — left in place for the owner to prune.
**Files:** `docs/07_tutorial5.md`, `static/img/tutorials/ink-modality-color.webp` (new), `ink-modality-infrared.webp` (new), `ink-modality-xray.webp` (new), `ink-signal-volumetric.webp` (new), `ink-iterative-labeling.webp` (new).

## C36 — landing: reveal-split hero, tighter margins, compact news (design workflow)

Owner brief: shrink the hero, convey "volcano → scroll → unwrap", compact the news strip, use desktop width better. Built from three parallel design-agent specs + a visual-critic iteration (all prototyped against the live build before implementation).
- **Hero 865px → 566px (desktop), 1085 → 910px (mobile)**: content-driven height (no more viewport filling); headline capped at 2 lines; copy+CTAs left, new **reveal card** right — plays /img/firstscroll/hero-reveal.webm ONCE on desktop (idle-deferred, data-saver-aware, first-frame poster swapped in pre-attach so playback is monotonic) and holds on its final frame; poster-only (29 KB end-frame webp = the payoff) on mobile/reduced-motion. Card is 21:9 (crops the footage's dead black + watermark). The 5-stat strip becomes a full-width hairline-topped band closing the hero (space-between on desktop; 1+2×2 grid on phones). Deleted: the old BREAKING card + duplicate "Read the breaking announcement" link (−233 KB eager images). 3D-model option evaluated and rejected (three.js on the critical path duplicating what the video shows).
- **Margins**: landing container 1152 → 1280px (40px gutters ≥997px), content 1088 → 1200px at 1440; story section media widens to 880px while prose keeps its 640px measure; awarded-prizes table aligns with the open board. Docs pages untouched; zero effect <997px.
- **News strip**: section padding 48→32px (24px mobile), rows 56→48px min-height, single-line ellipsis titles on desktop, single-line "title … date" ticker rows on phones (44px tap targets kept); the hardcoded "Get Started" cell removed — the strip is now 4 real posts. Desktop section 132→107px; mobile 313→217px.
- New assets: `hero-reveal-end-960.webp` (29 KB), `hero-reveal-start-960.webp` (7 KB), generated from hero-reveal.webm.
**Files:** `src/components/{Landing,LatestPosts}.js`, `src/css/{landing,chrome}.css`, `static/img/firstscroll/hero-reveal-{start,end}-960.webp` (new).

## C37 — hero polish, volcano repositioned, TOC beside content, tutorials under Open Problems

- **Hero**: intro cut to one sentence ("…reading the carbonized Herculaneum scrolls without opening them."); title clamp 2.25–3rem → 2–2.625rem, tagline and reveal caption a step smaller. **Volcano moved below the text**: scrim flipped from left-solid/right-visible (the reveal card was covering the volcano zone) to top-solid/bottom-visible — the eruption glow now rises along the hero's lower band on desktop and mobile.
- **News strip**: title 14→13px, meta 11→10px.
- **"Edit this page" removed site-wide** (docs `editUrl` dropped).
- **Docs TOC now sits beside the main column**: `--ifm-container-width` 1100→1000px; TOC gap to the article 89px → 16px at 1440.
- **Tutorials restructure**: the outdated tutorials hub (`/tutorial`) moved to Archive (URL alive); the six tutorials now nest as a collapsed expandable under **Open Problems** in the sidebar (category link = the post); the post gains a "🎓 Related tutorials" section with a stage-by-stage expandable list; the navbar "Tutorials" item removed (Open Problems is the entry point).
- Verified at 360/1024/1440/1920: zero horizontal overflow everywhere; build green (link checker).
**Files:** `src/components/Landing.js`, `src/css/{landing,chrome}.css`, `docusaurus.config.js`, `sidebars.js`, `docs/37_2026_open_problems.md`.

## C38 — funnel rework + live review round (owner session)

- **Funnel (A+C)**: Open problems moved above Open prizes (desktop offset 1,505 → 824px; mobile 2,171 → ~1,100px); gap after the news strip halved (48/32px); hero pool stat now scrolls to the on-page board.
- **Hero**: intro trimmed ("…without opening them."); "after 275 years" and "sealed since 79 AD" clauses cut; "Vesuvius Challenge now moves" → "The challenge now moves"; headline gap 44 → 32px desktop / 32 → 20px mobile; **$1M Grand Prize countdown chip** top-right over the volcano (client-computed days to 2027-06-25, links prizes#2027-grand-prize; inline pill above the headline on mobile).
- **Docs chrome**: TOC is a surface card with a bigger "ON THIS PAGE" header (13px); prizes page's inline TOCInline removed; TOCs restored on Get Started + Curated Datasets; story/sponsor expandable arrows 0.75 → 1.25rem and brighter.
- **Responsive audit**: 7 routes × 11 widths (320–1920) zero horizontal overflow. Two real bugs found and fixed: the doc column now yields to the fixed 220px TOC at 997–1150px, and `main` gets `min-width: 0` (the no-wrap pipeline strip was propagating a 752px intrinsic minimum past the viewport at 997px).
- **Navbar**: "Open Problems" → "Problems".
- **Data browser**: patches count removed from grid cards too; "first Latin in progress" and "3 scans at sub-2µm…" subtexts dropped (the latter required removing the tile's `derive: resolution` so buildIndex doesn't regenerate it).
**Files:** `src/components/{Landing,LatestPosts}.js`, `src/components/atlas/ScrollCard.js`, `src/css/{landing,chrome}.css`, `docusaurus.config.js`, `docs/{34_prizes,01_get_started,02_data_datasets}.md`, `src/data/atlasOverlay.json`.

## C39 — "Ask the Scrolls": grounded Q&A chat assistant (multi-agent build)

**Why:** visitors ask questions daily that the site already answers (prize rules, data licenses, how to start). A lightweight assistant grounded in **all site content** deflects those and makes the site more useful. Research phase (3-agent web workflow + live pricing check) picked build-over-buy: full-corpus-in-prompt beats RAG at this corpus size (~453KB ≈ 110k tokens), and the Vercel AI Gateway (zero-markup, one key, any `provider/model` string) satisfies the provider-flexible requirement — default `openai/gpt-5-mini` (~$0.004/question cached, switchable via `CHAT_MODEL` env, no code change).

- **Corpus pipeline** `scripts/genChatCorpus.js`: parses all 33 docs (strips JSX/images, keeps prose/tables/KaTeX, converts admonitions, root-anchors doc-relative links so the bot never emits a path that breaks on nested routes) + a synthetic Data Browser catalog page (per-scroll pixel sizes, licenses, PHerc. Paris 4 legal notice verbatim). Emits `static/llms-full.txt` (public, llms.txt convention) + `api/_lib/corpus.json` (both gitignored). Runs **after** `docusaurus build` so routes validate against the fresh sitemap (skip-with-warning when absent in dev); byte-deterministic; self-checks (anchor facts, size band, every URL in sitemap) fail the build loudly.
- **Endpoint** `api/chat.mjs` (+ `api/_lib/handler.mjs`, `rateLimit.mjs`): Vercel Function, AI SDK 7 `streamText` via AI Gateway with `caching: 'auto'` (the byte-stable system prompt = guardrails + full corpus caches provider-side), plain-text streaming response. Origin allowlist, Upstash sliding-window rate limit (in-memory fallback, fail-open), input caps (≤8 msgs, ≤1500 chars), 60s abort, per-request JSON usage logging (incl. cached tokens). `CHAT_MOCK=1` mode + `scripts/devChatServer.mjs` shim for keyless local testing. Guardrails: answer only from corpus, cite pages as path-only markdown links, flag archived content, decline off-topic, resist prompt injection, reply in the user's language.
- **Widget** `src/theme/Root.js` + `src/components/ChatWidget/*` + `src/css/chat.css`: "Ask" trigger (backdrop-blur surface pill, z-170, steps above the landing prize sticker on mobile), panel lazy-loaded on first open (0 initial JS), desktop 380×560 card / mobile 100dvh sheet with safe-area insets, streaming render with Stop, safe markdown-lite renderer (React elements only, scheme-checked links, Docusaurus `Link` for internal citations), sessionStorage persistence across nav/reload, dialog a11y (focus trap on mobile, Escape via document-level listener, aria-live once per completed message), example-question chips, disclaimer line.
- **Config:** `docusaurus.config.js` gains `customFields.chatEndpoint` (env `CHAT_ENDPOINT`, default `/api/chat`, empty disables widget) + `chat.css`; `vercel.json` gains `functions.api/chat.mjs.maxDuration=60`. New deps: `ai@7`, `@upstash/ratelimit`, `@upstash/redis`, `zod`. **Deploy needs env vars:** `AI_GATEWAY_API_KEY` (required), optional `CHAT_MODEL`, `UPSTASH_REDIS_REST_URL/TOKEN`, `CHAT_ALLOWED_ORIGINS`.
- **QA:** build green with corpus self-checks; 124 Playwright checks across 3 routes × 4 viewports (320–1440): zero horizontal overflow with panel open/closed, ≥44px targets, zero console errors, streaming/nav/restore flows, footer links clickable at 320px beside the trigger. One bug found & fixed: Escape-to-close died after the panel's own citation link navigated (Docusaurus moves focus to `<body>`; panel-scoped keydown never fired) → document-level listener while open. Also fixed: capped histories arriving assistant-first (rejected by Anthropic models) now trimmed client- and server-side. 10-question eval harness (incl. injection + off-topic probes + dead-link check on every emitted URL) runs in mock mode; live run pending `AI_GATEWAY_API_KEY`. Visual critic verdict SHIP; its polish items applied: footer bottom clearance ≤996px (trigger permanently occluded the copyright line), no input autofocus on touch devices (soft keyboard covered the suggestion chips — panel receives dialog focus instead), empty-state mark+hint grouped as one block, disclaimer shortened ("Answers are AI-generated from site content.").

**Files:** `scripts/{genChatCorpus.js,devChatServer.mjs}`, `api/chat.mjs`, `api/_lib/{handler,rateLimit}.mjs`, `src/theme/Root.js`, `src/components/ChatWidget/{index,ChatPanel,MarkdownLite}.js`, `src/css/chat.css`, `docusaurus.config.js`, `vercel.json`, `package.json`, `.gitignore`.

## C40 — chat assistant: owner review round

- **Getting-started guidance**: the system prompt (and the mock answer) now recommends a canonical path — [get started](/get_started) → [open problems](/2026_open_problems) → tutorials ([spiral fitting](/tutorial_spiral), [ink detection](/tutorial5)) → join Discord.
- **Empty state**: "Can I use the scroll data commercially?" chip removed (2 chips remain); hint now reads "…answers come from this site's content and may be out of date. Join the Discord community for the latest information!" with a live Discord link (hint links get the body-link grammar).
- **Trigger**: much larger on desktop ≥997px (56px tall, 1rem label, 20px glyph, 28px inset); mobile keeps the compact 44px pill.
**Files:** `api/_lib/handler.mjs`, `src/components/ChatWidget/ChatPanel.js`, `src/css/chat.css`.

## C41 — chat endpoint: fix crash on Vercel + preview-domain origins

**Why:** first remote deploy crashed with `request.headers.get is not a function` — Vercel's Node.js runtime invokes the default export classic-style (`handler(req, res)` with an IncomingMessage), not with a Web `Request`. And the origin allowlist only contained `https://scrollprize.org`, which would have 403'd every Vercel preview domain (browsers send `Origin` on every POST, same-origin included).
- `api/chat.mjs` is now a dual-mode entry: bridges Node `(req, res)` ↔ Web Request/Response (same adapter the dev shim uses, streaming preserved via `Readable.fromWeb`), still handles a Web `Request` directly if the runtime passes one.
- `resolveCors` gains a same-host rule: an `Origin` whose host equals the request's own `x-forwarded-host`/`host` is always allowed — production and preview domains both work with zero config.
- Verified locally under the classic signature: same-origin POST streams 200, foreign origin 403, preflight 204, bad body 400.
**Files:** `api/chat.mjs`, `api/_lib/handler.mjs`.

## C42 — "Virtual Philodemus": prompt v2, section-anchor citations, assistant identity

**Why:** owner prompt-tuning round — better answers, more precise citations, and a persona that makes the widget feel like talking to a real assistant.
- **Prompt v2** (`api/_lib/handler.mjs`): restructured into VOICE / GROUNDING / CITATIONS / GUIDANCE / BOUNDARIES. New rules: go technically deep for technical questions (methods, architectures, resolutions, formats); never compute relative time — the model doesn't know today's date, so deadlines are stated as absolute dates only; licensing/data-use answers (esp. PHerc. Paris 4) restate the license/notice faithfully, no loose paraphrase, no legal advice; partial answers instead of all-or-nothing; 1–3 links woven in, not link dumps; archived pages flagged inline before/at the link with a pointer to the superseding page; vague questions get a brief best answer + one clarifying question; self-identity rule (AI assistant, never claims to be human or the historical Philodemus).
- **Section-anchor citations**: `genChatCorpus.js` extracts every h2–h4 anchor id from the built HTML (runs post-build, so anchors are verified against what Docusaurus actually generated — no dead anchors possible) and annotates corpus headings as `[section: /page#anchor]`. 303/303 headings matched, incl. custom ids like `/prizes#first-title-prize`. The prompt instructs citing the most specific section.
- **Identity**: renamed to **Virtual Philodemus** (panel title + subtitle "Vesuvius Challenge assistant", aria labels). New `ChatAvatar` mark — a drawn Archimedean spiral (the scroll CT cross-section, the project's most recognizable visual) as inline SVG, zero image payload: ember-on-raised circular chip in the header, large quiet version in the empty state, ember mark in the trigger, whose desktop label is now "Ask Philodemus" ("Ask" on mobile). Empty-state hint introduces the persona.
**Files:** `api/_lib/handler.mjs`, `scripts/genChatCorpus.js`, `src/components/ChatWidget/{index,ChatPanel,ChatAvatar}.js`, `src/css/chat.css`.

## C43 — chat: low reasoning effort, rate limit 30/10min

- Model stays `openai/gpt-5-mini`; reasoning effort now `low` by default (`providerOptions.openai.reasoningEffort`, env-overridable via `CHAT_REASONING_EFFORT`) — grounded Q&A needs retrieval-and-phrasing, not deep reasoning; cuts latency and reasoning-token spend. Ignored by non-OpenAI models.
- Default per-IP rate limit raised 10 → **30 requests / 10 minutes** (both Upstash and in-memory paths); daily cap raised 60 → 150 so the burst allowance stays meaningful across a day.
**Files:** `api/_lib/handler.mjs`, `api/_lib/rateLimit.mjs`.

## C44 — Open Problems restyle: figure system, Philodemus callout, team context (multi-agent)

**Why:** the article's 26 images rendered as a one-size-fits-all full-width parade — white matplotlib figures glaring on the dark theme, comparisons stacked 700px apart, centered-italic caption walls, two captions silently unbound from their images. Owner also wanted a Last-updated stamp, an "ask Philodemus" pointer for overwhelmed readers, and the team in the chat corpus.
- **Figure system** (Fable vision critique → implementation): new `src/components/Figure.js` (variants full/medium/pair/stack + `card`/`enlarge` modifiers; self-stamps width/height from the image manifest; caption strings render markdown links safely) + page-agnostic `.vc-fig*` CSS in chrome.css replacing the old adjacency-caption rule. All 26 images migrated to 22 figures per the critique's mapping: white figures on `--vc-surface` plate cards with brightness knockdown, comparisons as pair rows (micro-captions "DLS, 7.91 µm"/"ESRF, 2.4 µm") or grouped stacks with one caption (fixing both mis-bound caption pairs), VC3D walkthrough at 560px medium, dense grids click-to-enlarge, full-width reserved for the ~6 dark-native heroes. Captions left-hung 0.8125rem/60ch; figure rhythm 32px. Long captions' explanatory tails promoted to body text (content preserved verbatim).
- **Date stamp**: `*Last updated: July 10, 2026*` (spiral-tutorial idiom); the old "July 2026" publish div removed (critic: double date).
- **Philodemus callout**: `src/components/ChatWidget/ChatCallout.js` after the pipeline strip — surface card, ember rule, avatar, "Ask Philodemus" button dispatching new `vc:open-chat` CustomEvent; ChatWidget/ChatPanel gained the listener + prefill plumbing (draft prefilled, never auto-sent).
- **Chat corpus**: `genChatCorpus.js` now extracts `<Figure>` captions (22/22 preserved — previously ALL image captions were dropped from the corpus); new synthetic "Vesuvius Challenge Team" section (`/#team`, anchor-verified) merging landing team groups with per-person roles from `src/data/teamRoles.json` (curated from arXiv:2606.29085 Author Contributions, cited); new self-check needles. Corpus +9k chars, byte-deterministic.
- **Mobile**: TutorialsTop pipeline strip 2×2 below 768px (CSS-only, arrows hidden — was wrapping 3+1); pairs/stacks collapse to 1 col; portraits capped; desktop TOC max-height now clears the fixed chat trigger (was occluding its last entries at 1440×900).
- **QA**: build green; 20 scripted checks (overflow 320–1440, console-clean, 22 figures with dims, enlarge-bypasses-lightbox ×3, lightbox open/close, callout ≥44px + prefill-unsent, strip 2×2, regressions on / /faq /firstscroll); 109 tiles captured; final Fable sign-off **SHIP** (its POLISH + date NIT applied; two "pair" nits were baked single-image composites — unfixable without editing images, out of scope).
**Files:** `docs/37_2026_open_problems.md`, `src/components/Figure.js`, `src/components/ChatWidget/{ChatCallout,index,ChatPanel}.js`, `src/css/{chrome,chat}.css`, `scripts/genChatCorpus.js`, `src/data/teamRoles.json`.

## C45 — open problems: owner review round

- All figure captions and pair micro-captions centered (were left-hung per the critique; owner preference wins).
- "Project & further reading" reference subsection removed (site/GitHub/HF/Substack links — all cited inline in the piece anyway; no inbound anchors).
- ID11 fiber cross-sections (image2+3) switched from grouped stack to side-by-side pair.
**Files:** `docs/37_2026_open_problems.md`, `src/css/chrome.css`.

## C46 — open problems: "Open problem" takeaway boxes

**Why:** owner wants each article section's core problem statement to hit at a glance — starting from the ink-recovery caveat ("recovered PHerc. 1667's text, but not guaranteed everywhere"), which deserved a box of its own.
- New admonition variant `.vc-problem` (chrome.css): same family as the help callouts, but ember label + icon, 3px edge, faint accent wash — unmistakably a problem statement.
- Four boxes, one per major section, each distilling that section's takeaway (owner-reviewed wording): §1 Scanning — compressed regions leave the scanner degraded, sub-micron whole-scroll scans impractical, limiting pipeline stage unknown scroll-by-scroll; §2 Unwrapping — no fully automatic complete trace; label quality one of the main bottlenecks; §3 Ink — the verbatim PHerc. 1667 caveat (model links kept as following prose); §4 Data scale — cloud-native or impractical; infrastructure determines what research is possible.
**Files:** `docs/37_2026_open_problems.md`, `src/css/chrome.css`.

## C50 — VC3D tutorial promoted to current

**Why:** David's new draft VC3D tutorial (`06_tutorial_VC3D.md`, added in `0f344e474` under Archive) is the live guide for the project's main tool — it belongs with the current tutorials.
- Sidebar: moved from Archive into Overview → Open Problems, first (above Spiral Fitting and Ink Detection).
- Date stamp `*Last updated: July 11, 2026*` added to the page (site convention).
- Open Problems: listed first under "Related tutorials → Up to date" (marked early draft); inline link added where the article introduces VC3D as the visualization tool ("A hands-on VC3D tutorial walks through…").
- Landing: the Virtual Unwrapping challenge box's CTA ("Chart the Path") now leads to /tutorial_VC3D — mirroring how Ink Detection's "Find a Letter" leads to /tutorial5 (its title still deep-links the article section).
**Files:** `sidebars.js`, `docs/06_tutorial_VC3D.md`, `docs/37_2026_open_problems.md`, `src/components/Landing.js`.

## C52 — tutorials: no archived links from current pages, Philodemus callouts, deprecated repo link removed

- **TutorialsTop defaults** pointed at archived pages (/tutorial1, /tutorial2, /segmentation) — the strips atop Spiral Fitting and Ink Detection sent readers to ARCHIVED content. Defaults now current-only: Scanning → /2026_open_problems#1-scanning…, Representation → /tutorial_VC3D, Segmentation and Flattening → /tutorial_spiral, Ink → /tutorial5 (unchanged). Explicit `links` overrides (Open Problems' strip) unaffected.
- **In-prose archived links retargeted** on the same pages: spiral's "manual segmentation in VC3D" and tutorial5's "segmented" now → /tutorial_VC3D. Verified zero links to archived pages remain on any current tutorial.
- **Philodemus callouts** added to all three tutorials with page-specific prefills (VC3D: "Help me get started with VC3D"; spiral/ink: "Walk me through the … tutorial").
- **Deprecated data repo link removed** from Data Formats (`https://data.aws.ash2txt.org/samples/`) — docs-only scope per owner: the 10 Neuroglancer viewer links whose zarr source is that host stay (those DLS volumes aren't mirrored in the open-data bucket yet; removing them would kill data access for PHerc0332/1667/Paris4's older scans).
**Files:** `src/components/TutorialsTop.js`, `docs/{38_tutorial_spiral,07_tutorial5,06_tutorial_VC3D,02_data}.md`.

## C53 — data browser: drop old-bucket "Full dataset" rows

**Why:** owner spotted `Full dataset (HTTP) -> data.aws.ash2txt.org/samples/...` on detail pages after C52's deprecated-link removal. Those two rows were hardcoded to the old `s3://vesuvius-challenge/` bucket and had become exact duplicates of the "HTTP"/"S3" rows above them, which already point at the same tree in the open-data bucket (verified per-scroll prefixes exist). Removed both rows (+ dead `toHttp` import). Remaining old-host references: only inside Neuroglancer viewer URLs for the not-yet-mirrored DLS volumes (kept deliberately).
**Files:** `src/components/atlas/DataCatalog.js`.

## C54 — prizes: eligible scroll list corrected to 14, full volume names

**Why:** owner narrowed the 2027 Grand Prize eligible list to 14 scrolls (dropping 175A/B, 306B, 483A/B, 490A/B, 846A/B) and wants each entry to show the full volume name — timestamp, voxel size, energy — exactly as the Data Browser displays it.
- Eligible list rebuilt programmatically from the freshly regenerated atlas index: visible text is now the scan name (e.g. `20250720091415-9.362um-1.2m-113keV`) instead of just the voxel size; each entry's existing Neuroglancer href verified to match the live volume zarr path (all 14 matched — the scan *name* timestamp differs from the zarr *path* timestamp by design: acquisition vs processing).
- Count updated 23 → 14 in the details summary, the prizes frontmatter hook ("Fully unroll and read one of 14 sealed scrolls." — flows to the landing prize board via prizes-data), and get_started's Grand Prize line.
**Files:** `docs/34_prizes.md`, `docs/01_get_started.md`.

## C55 — prizes: GP criteria precision (owner wording)

- "Valid explanation" for missing ink now defined: valid only if the Vesuvius Challenge Team acknowledges it as valid.
- Submission package gains a leading **Meshes** requirement: complete tifxyz set covering the whole scroll surface, each with a low-distortion isometric 2D parametrization (flattening) included.
- Images: must be generated programmatically from the reconstructed CT volume AND the corresponding submitted mesh; each image named after the tifxyz mesh it was rendered from (replaces the vaguer position-description bullet). First Letters/Title sections untouched (mesh package is GP-specific).
**Files:** `docs/34_prizes.md`.

## Baseline (pre-restyle, recorded 2026-07-03)

- `yarn build` green; 82 sitemap routes all HTTP 200.
- Landing (desktop, full scroll-flush): **15,248,955 bytes / 137 requests**.
- `build/` 624,524 KB; `build/assets` 22,824 KB.
- Console errors: 20 (all the C0 404). Mobile: no horizontal overflow; ~10 sub-44px tap targets per page (hamburger 30×30, footer links 29–35×32).
- 72 arbitrary-value Tailwind classes inventoried across docs (purge tripwire baseline); 1 pre-existing purge victim (`w-[70%]`, mdx-only) recorded.
