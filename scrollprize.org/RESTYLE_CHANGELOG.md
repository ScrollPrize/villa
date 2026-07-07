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

## Baseline (pre-restyle, recorded 2026-07-03)

- `yarn build` green; 82 sitemap routes all HTTP 200.
- Landing (desktop, full scroll-flush): **15,248,955 bytes / 137 requests**.
- `build/` 624,524 KB; `build/assets` 22,824 KB.
- Console errors: 20 (all the C0 404). Mobile: no horizontal overflow; ~10 sub-44px tap targets per page (hamburger 30×30, footer links 29–35×32).
- 72 arbitrary-value Tailwind classes inventoried across docs (purge tripwire baseline); 1 pre-existing purge victim (`w-[70%]`, mdx-only) recorded.
