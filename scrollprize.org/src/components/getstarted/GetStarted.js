import React from "react";
import Heading from "@theme/Heading";
import { usePluginData } from "@docusaurus/useGlobalData";
import AwardedTotal from "../AwardedTotal";
import InkDemo from "./InkDemo";
import SegDemo from "./SegDemo";
import { useCoarsePointer } from "./demoUtils";

/*
 * /get_started — landing-class onboarding page.
 * Structure: hero -> READ (ink demo) -> UNWRAP (seg demo + spiral pointer)
 * -> the real thing -> pick your path (geometry first) -> proof -> close.
 * The page IS the pipeline: the visitor does, in miniature, what the
 * challenge does at scale — then gets routed by what they enjoyed.
 */

const usd = new Intl.NumberFormat("en-US", {
  style: "currency",
  currency: "USD",
  maximumFractionDigits: 0,
});

function SectionHead({ kicker, title, id, children }) {
  return (
    <header className="vc-gs-sechead" id={id}>
      <p className="vc-kicker">{kicker}</p>
      <Heading as="h2" className="vc-gs-h2">
        {title}
      </Heading>
      {children}
    </header>
  );
}

export function GetStarted() {
  const { prizes = [] } = usePluginData("prizes-data") || {};
  const { counts = {} } = usePluginData("atlas-data") || {};
  const openPool = prizes.reduce((s, p) => s + p.amount, 0);
  const scanned = (counts.scrolls ?? 35) + (counts.fragments ?? 10);
  const coarse = useCoarsePointer(); // "tap" vs "click" in demo copy

  return (
    <main className="vc-gs">
      {/* ================= hero ================= */}
      <section className="vc-gs-hero" aria-labelledby="gs-title">
        {/* the unwrapped PHerc. 1667 surface as a full-bleed backdrop
            (middle ~third of the strip, zoomed), landing-page style: a
            scrim keeps the text column solid and fades the papyrus in
            toward the right */}
        <div className="vc-gs-hero__bg" aria-hidden="true" />
        <div className="vc-gs-hero__scrim" aria-hidden="true" />
        <div className="vc-gs-container vc-gs-hero__content">
          <p className="vc-kicker">Vesuvius Challenge · Get started</p>
          <Heading as="h1" id="gs-title" className="vc-gs-h1">
            One scroll read.
            <br />You could read the next.
          </Heading>
          <p className="vc-gs-lead">
            The <a href="/firstscroll">first scroll has been read</a>, and the
            people who read it were newcomers not long before: students,
            engineers, tinkerers who found this project and stayed. Hundreds
            of scrolls remain sealed
            {openPool > 0 ? (
              <>
                , and <strong>{usd.format(openPool)}</strong> in prizes is on
                the table for whoever helps open them. Come claim a piece of
                it.
              </>
            ) : (
              "."
            )}
          </p>
          <p className="vc-gs-lead vc-gs-lead--em">
            Here's what the work actually looks like, where you'd fit in it,
            and what to do first.
          </p>
          <div className="vc-gs-hero__ctas">
            <a className="vc-btn-outline" href="https://discord.gg/V4fJhvtaQn">
              Join Discord
            </a>
          </div>
        </div>
      </section>

      {/* ================= part 1: read ================= */}
      <section className="vc-gs-section" aria-labelledby="read">
        <div className="vc-gs-container">
          <SectionHead kicker="Part 1 · Read" title="Can you find the ink?" id="read">
            <p className="vc-gs-sub">Train your first ink model.</p>
            <p className="vc-gs-body">
              The ink is carbon. So is the scorched papyrus underneath it.
              To an X-ray the two look nearly identical. But one scroll's ink
              cracked as it dried, and a person spotted the faint texture by
              eye. Those cracks gave up the
              first words ever read from inside an unopened scroll. Most
              scrolls give away less, so models have since learned to catch
              traces that can slip past even the eye.
            </p>
            <p className="vc-gs-body">
              This is easier to try than to explain, so here's a piece
              of that same scroll. Paint the ink you can see, and a model
              trains on your labels to predict the rest. It's only as good
              as the labels you give it.
            </p>
          </SectionHead>
          <InkDemo />
          <nav className="vc-gs-next" aria-label="Go deeper: ink detection">
            <span className="vc-gs-dim">Go deeper:</span>
            <a href="/tutorial5">Ink detection tutorial: train the model</a>
            <a href="/prizes#first-letters-prizes">
              First Letters: up to $500,000
            </a>
            <a href="/2026_open_problems#3-ink-recovery-reading-the-scrolls">
              The open problem
            </a>
          </nav>
        </div>
      </section>

      {/* ================= part 2: unwrap ================= */}
      <section className="vc-gs-section" aria-labelledby="unwrap">
        <div className="vc-gs-container">
          <SectionHead
            kicker="Part 2 · Unwrap"
            title="Where did that surface come from?"
            id="unwrap"
          >
            <p className="vc-gs-sub">Trace your first surface.</p>
            <p className="vc-gs-body">
              Now rewind one step. You just read ink off a flat surface, but
              nothing in a rolled scroll is flat. A CT scan gives us a
              billion voxels of a single sheet wound around itself hundreds
              of times, crushed and torn by the eruption. Before anyone can
              look for ink, that sheet has to be found, traced, and
              flattened. We call it <em>virtual unwrapping</em>.
            </p>
            <p className="vc-gs-body">
              Below is real data from PHerc. Paris 4, and the same sheet you
              read the word from in Part 1. {coarse ? "Tap" : "Click"} where
              the scroll pulses to open a cross-section. Every thin bright line in it is one layer of
              the rolled-up sheet. Follow a single layer with your dots and
              the flattened surface builds on the right. Drift onto a
              neighboring layer and it smears.
            </p>
          </SectionHead>
          <SegDemo />
          <nav className="vc-gs-next" aria-label="Go deeper: unwrapping">
            <span className="vc-gs-dim">Go deeper:</span>
            <a href="/tutorial_VC3D">VC3D tutorial: trace a segment</a>
            <a href="/tutorial_spiral">Spiral fitting tutorial</a>
            <a href="/prizes#2027-grand-prize">2027 Grand Prize: $1,000,000</a>
            <a href="/2026_open_problems#2-unwrapping-turning-disconnected-voxels-into-a-surface">
              The open problem
            </a>
          </nav>
        </div>
      </section>

      {/* ================= the real thing ================= */}
      <section className="vc-gs-section vc-gs-section--band" aria-labelledby="pipeline">
        <div className="vc-gs-container">
          <SectionHead
            kicker="The full pipeline"
            title="It works. It doesn't scale yet."
            id="pipeline"
          >
            <p className="vc-gs-body">
              Scan, unwrap, read. You've now done all three, just very
              small. At full size a scroll is several terabytes of CT data,
              a surface runs for meters through hundreds of turns, and
              parts of the tracing still need a human in the loop. The
              first scroll settled whether any of this is possible. The
              prizes are for scaling the methods up, so the entire library
              can finally be read.
            </p>
          </SectionHead>
          <div className="vc-stat-strip vc-gs-stats">
            <a className="vc-stat vc-stat--link" href="/data_browser">
              <span className="vc-stat__value">{scanned}</span>
              <span className="vc-stat__label">scrolls & fragments scanned</span>
            </a>
            <a className="vc-stat vc-stat--link" href="/data_browser">
              <span className="vc-stat__value">12</span>
              <span className="vc-stat__label">in segmentation</span>
            </a>
            <a className="vc-stat vc-stat--link" href="/data_browser">
              <span className="vc-stat__value">4</span>
              <span className="vc-stat__label">with recovered text</span>
            </a>
            {openPool > 0 && (
              <a className="vc-stat vc-stat--link" href="/prizes">
                <span className="vc-stat__value">{usd.format(openPool)}</span>
                <span className="vc-stat__label">open prize pool</span>
              </a>
            )}
          </div>
          <p className="vc-gs-body vc-gs-body--center">
            <a href="/2026_open_problems">Open Problems</a> lists what's
            still unsolved, and how to start on each one. The Grand Prize
            deadline is{" "}
            <a href="/prizes#2027-grand-prize">
              <strong>June 25, 2027</strong>
            </a>
            .
          </p>
        </div>
      </section>

      {/* ================= paths ================= */}
      <section className="vc-gs-section" aria-labelledby="paths">
        <div className="vc-gs-container">
          <SectionHead kicker="Pick your path" title="Where do you fit?" id="paths" />
          <div className="vc-gs-paths">
            <div className="vc-card vc-gs-path">
              <h3>I think in geometry</h3>
              <p>
                You just traced one layer of a scroll by hand. A real
                scroll has hundreds, crushed together and torn, and reading
                the library means software that follows them on its own. It's a
                hard surface-reconstruction problem: meshes, optimization,
                3D vision.
              </p>
              <p className="vc-gs-path__prize">
                2027 Grand Prize · $1,000,000
              </p>
              <a className="vc-gs-path__cta" href="/2026_open_problems">Read the open problems →</a>
            </div>
            <div className="vc-card vc-gs-path">
              <h3>I train models</h3>
              <p>
                You just trained the toy version. Real ink detection works
                the same way: a computer-vision problem with 2,000-year-old
                labels, and the tutorial takes you from download to a first
                prediction in a weekend. Better predictions can win prizes
                and uncover more unseen text.
              </p>
              <p className="vc-gs-path__prize">
                First Letters · up to $500,000
              </p>
              <a className="vc-gs-path__cta" href="/tutorial5">Ink detection tutorial →</a>
            </div>
            <div className="vc-card vc-gs-path">
              <h3>I build tools</h3>
              <p>
                Most of the software here was written by people in the
                community: viewers, data loaders, training code. The best
                open-source contribution each month wins $20,000.
              </p>
              <p className="vc-gs-path__prize">
                Progress Prizes · $590,000 / year
              </p>
              <a className="vc-gs-path__cta" href="/community_projects">See the tools →</a>
            </div>
            <div className="vc-card vc-gs-path">
              <h3>I'm just curious</h3>
              <p>
                Everything here is open: the scans, the renders, the
                recovered text. All of it is a browser tab away, and following along
                closely is how many contributors started.
              </p>
              <p className="vc-gs-path__prize">No setup · no deadline</p>
              <a className="vc-gs-path__cta" href="/data_browser">Open the Data Browser →</a>
            </div>
          </div>
          <p className="vc-gs-body vc-gs-body--center vc-gs-dim">
            Still deciding? The technical frontier is in{" "}
            <a href="/2026_open_problems">Open Problems</a>, the data is all{" "}
            <a href="/data">open</a>, and the <a href="/faq">FAQ</a> covers
            the rest.
          </p>
        </div>
      </section>

      {/* ================= proof ================= */}
      <section className="vc-gs-section vc-gs-section--band" aria-labelledby="proof">
        <div className="vc-gs-container">
          <SectionHead kicker="Track record" title="Former winners were newcomers too." id="proof">
            <p className="vc-gs-body">
              The first word ever read from inside a sealed scroll was
              ΠΟΡΦΥΡΑϹ, <em>purple</em>, the word you uncovered in Part 1.
              It was found by a student who had joined the community only
              months earlier. The <AwardedTotal /> awarded since has gone
              mostly to individuals and small teams. Many of them first met
              on Discord.
            </p>
          </SectionHead>
          <p className="vc-gs-body vc-gs-body--center">
            <a href="/winners">Meet the winners →</a>
          </p>
        </div>
      </section>

      {/* ================= close ================= */}
      <section className="vc-gs-section vc-gs-close" aria-labelledby="community">
        <div className="vc-gs-container">
          <SectionHead
            kicker="Last step"
            title="Join the community."
            id="community"
          >
            <p className="vc-gs-body">
              Every path above starts in the same place. Come introduce
              yourself and ask anything: the Vesuvius Challenge team is around every day, and
              so are contributors who have hit every wall
              before you and are glad to help you past them.
            </p>
          </SectionHead>
          <div className="vc-gs-hero__ctas vc-gs-close__ctas">
            <a className="vc-btn" href="https://discord.gg/V4fJhvtaQn">
              Join the Discord
            </a>
            <a className="vc-btn-outline" href="https://scrollprize.substack.com">
              Get the newsletter
            </a>
          </div>
          <p className="vc-gs-dim vc-gs-body--center">
            Questions? <a href="mailto:team@scrollprize.org">team@scrollprize.org</a>
          </p>
        </div>
      </section>
    </main>
  );
}
