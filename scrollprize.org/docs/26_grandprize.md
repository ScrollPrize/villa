---
title: "Vesuvius Challenge 2023 Grand Prize awarded: we can read the scrolls!"
sidebar_label: "2023 Grand Prize"
hide_table_of_contents: true
hide_title: true
---

<head>
  <html data-theme="dark" />

  <meta name="description" content="The 2000-year-old scroll discusses music, food, and how to enjoy life’s pleasures." />

  <meta property="og:type" content="website" />
  <meta property="og:url" content="https://scrollprize.org/grandprize" />
  <meta property="og:title" content="Vesuvius Challenge 2023 Grand Prize awarded: we can read the scrolls!" />
  <meta property="og:description" content="The 2000-year-old scroll discusses music, food, and how to enjoy life’s pleasures." />
  <meta property="og:image" content="https://scrollprize.org/img/social/opengraph.jpg" />

  <meta property="twitter:card" content="summary_large_image" />
  <meta property="twitter:url" content="https://scrollprize.org/grandprize" />
  <meta property="twitter:title" content="Vesuvius Challenge 2023 Grand Prize awarded: we can read the scrolls!" />
  <meta property="twitter:description" content="The 2000-year-old scroll discusses music, food, and how to enjoy life’s pleasures." />
  <meta property="twitter:image" content="https://scrollprize.org/img/social/opengraph.jpg" />
</head>

<h1 className="color-white text-4xl md:text-7xl font-black !mb-2 leading-none tracking-tighter">Vesuvius Challenge 2023 Grand Prize awarded: <span style={{
  background:
    "radial-gradient(53.44% 245.78% at 13.64% 46.56%, #F5653F 0%, #D53A17 100%)",
  WebkitBackgroundClip: "text",
  WebkitTextFillColor: "transparent",
  backgroundClip: "text",
  textFillColor: "transparent",
}}>we can read the first scroll!</span></h1>

<div className="md:text-2xl text-lg font-medium mt-6 mb-2 opacity-80 leading-none tracking-tight">The 2000-year-old scroll discusses music, food, and how to enjoy life’s pleasures.</div>

<div className="opacity-60 mb-8 italic">February 5th, 2024</div>

We’re announcing the winners of the Vesuvius Challenge 2023 Grand Prize. We’ll look at how they did it, what the scrolls say, and what comes next.

*Join us for a celebration at the Getty Villa Museum in Los Angeles on March 16th, 4pm. <a href="https://www.getty.edu/visit/cal/events/ev_4074.html">More information here</a>.*



## Victory

Two thousand years ago, a volcanic eruption buried an ancient library of papyrus scrolls now known as the Herculaneum Papyri.

<figure className="">
  <video autoPlay playsInline loop muted className="w-[100%] sm:w-[55.6%]" poster="/img/grandprize/library-lava-small.webp">
    <source src="/img/grandprize/library-lava-small.webm" type="video/webm"/>
  </video>
  <video autoPlay playsInline loop muted className="hidden sm:inline sm:w-[44%]" poster="/img/grandprize/scroll-lava-small.webp">
    <source src="/img/grandprize/scroll-lava-small.webm" type="video/webm"/>
  </video>
  <figcaption className="mt-0">The scrolls were <a href="https://twitter.com/natfriedman/status/1703422593670541437">carbonized</a> by the eruption of Mount Vesuvius in 79 AD.</figcaption>
</figure>

In the 18th century the scrolls were discovered. Hundreds of them are now stored in a library in Naples, Italy; these lumps of carbonized ash cannot be opened without severely damaging them. But how can we read them if they remain rolled up?

<div className="flex flex-wrap mb-4">
  <div className="w-[50.5%] mr-4 mb-2"><img src="/img/grandprize/scroll-1-scale.webp" className="w-[100%]"/><figcaption className="mt-[-6px]">The scroll read by the winners.</figcaption></div>
  <div className="w-[45%]"><img src="/img/grandprize/unrolled-orig.webp" className="w-[100%]"/><figcaption className="mt-[-6px]">Result of an attempt to physically unroll a scroll.</figcaption></div>
</div>

On March 15th, 2023, Nat Friedman, Daniel Gross, and Brent Seales launched the <a href="https://scrollprize.org">Vesuvius Challenge</a> to answer this question. Scrolls from the Institut de France were imaged at the Diamond Light Source particle accelerator near Oxford. We released these high-resolution CT scans of the scrolls, and we offered more than \$1M in prizes, put forward by many generous donors.

<div className="flex flex-wrap mb-4">
  <div className="w-[41%] mr-4 mb-2"><img src="/img/grandprize/seth-diamond.webp" className="w-[100%]"/><figcaption className="mt-[-6px]">Seth Parker <a href="https://scrollprize.substack.com/p/new-scans-of-herculaneum-papyri-at">scanning</a> a scroll at the <a href="https://www.diamond.ac.uk/">Diamond Light Source</a> particle accelerator.</figcaption></div>
  <div className="w-[54%]">
    <video autoPlay playsInline loop muted className="w-[100%]" poster="/img/tutorials/scanning2.webp">
      <source src="/img/tutorials/scanning2.webm" type="video/webm"/>
    </video>
    <figcaption className="mt-[-6px]">Artistic visualization of constructing a 3D volume.</figcaption>
  </div>
</div>

A global community of competitors and collaborators assembled to crack the problem with computer vision, machine learning, and hard work.

Less than a year later, in December 2023, they succeeded. Finally, after 275 years, we can begin to read the scrolls:

<div className="mb-4">
  <a target="_blank" href="/img/grandprize/text_bcb-smaller.webp"><img src="/img/grandprize/text_bcb-smaller.webp" className="w-[100%]"/></a>
  <figcaption className="mt-[-6px]">Text from PHerc.Paris. 4 (Institut de France), unseen for 2,000 years. Roughly 95% of the scroll remains to be read.</figcaption>
</div>

The thoughts of our ancestors, locked in mud and ash for 2000 years, hidden in darkness — now, with the light of a worldwide effort shining upon them, finally seen again.

## Grand Prize

We received many excellent submissions for the Vesuvius Challenge Grand Prize, several in the final minutes before the midnight deadline on January 1st.

We presented these submissions to the review team, and they were met with widespread amazement. We spent the month of January carefully reviewing all submissions. Our team of eminent papyrologists worked day and night to review 15 columns of text in anonymized submissions, while the technical team audited and reproduced the submitted code and methods.

There was one submission that stood out clearly from the rest. Working independently, each member of our team of papyrologists recovered more text from this submission than any other. Remarkably, the entry achieved the criteria we set when announcing Vesuvius Challenge in March: 4 passages of 140 characters each, with at least 85% of characters recoverable. This was not a given: most of us on the organizing team assigned a less than 30% probability of success when we announced these criteria! And in addition, the submission includes another 11 (!) columns of text — more than 2000 characters total.

The results of this review were clear and unanimous: the Vesuvius Challenge Grand Prize of \$700,000 is awarded to a team of three for their excellent submission. Congratulations to **Youssef Nader, Luke Farritor, and Julian Schilliger**!


<div className="flex flex-wrap mb-4">
  <div className="w-[45%] sm:w-[31%] mr-4 mb-2"><img src="/img/grandprize/youssef-smaller.webp" className="w-[100%]"/><figcaption className="mt-[-6px]">Youssef Nader</figcaption></div>
  <div className="w-[45%] sm:w-[31%] mr-4 mb-2"><img src="/img/grandprize/luke-smaller.webp" className="w-[100%]"/><figcaption className="mt-[-6px]">Luke Farritor</figcaption></div>
  <div className="w-[45%] sm:w-[31%]"><img src="/img/grandprize/julian-smaller.webp" className="w-[100%]"/><figcaption className="mt-[-6px]">Julian Schilliger</figcaption></div>
</div>

All three winning team members have been strong community contributors since the very beginning of Vesuvius Challenge. You may remember Youssef. He is the Egyptian PhD student in Berlin who was able to read a few columns of text back in October, winning the second-place [First Letters Prize](firstletters). His results back then were particularly clear and readable, which made him the natural lead for the team that formed.

You might remember Luke as well: he is the 21-year-old college student and SpaceX intern from Nebraska, who was the first person in history to read an entire word from the inside of a Herculaneum scroll (ΠΟΡΦΥΡΑϹ, “purple”). This won him the first-place First Letters Prize, a few weeks before Youssef’s results.

And finally, you might remember Julian. He is the Swiss robotics student at ETH Zürich, who won three Segmentation Tooling prizes for his incredible work on Volume Cartographer. This enabled the 3d-mapping of the papyrus areas you see before you.

For the Grand Prize, they assembled into a superteam, crushing it by creating what was unanimously deemed the most readable submission.

The submission contains results from three different model architectures, each supporting the findings of the others, with the strongest images often coming from a [TimeSformer](https://arxiv.org/abs/2102.05095)-based model. Multiple measures prevent overfitting and hallucination, including results from multiple architectures, a study across input/output window sizes, label smoothing, and varying validation folds. Like with all our prizes, this ink detection code has been made public as open source (on [GitHub](https://github.com/younader/Vesuvius-Grandprize-Winner)), leveling up everyone in the community.

<div className="mb-4">
  <a target="_blank" href="/img/grandprize/youssef_text_wbb.webp"><img src="/img/grandprize/youssef_text_wbb-smaller.webp" className="w-[100%]"/></a>
  <figcaption className="mt-[-6px]">The winners’ main submission image (TimeSformer 64x64).</figcaption>
</div>

In addition to unparalleled ink detection, the winning submission contained the strongest auto-segmentation approach we have seen to date (more about the process of “segmentation” below). [ThaumatoAnakalyptor](https://github.com/schillij95/ThaumatoAnakalyptor) (roughly: Miracle Uncoverer) by Julian generates massive papyrus segments from multiple scrolls. Re-segmentations of well known areas validate previous ink findings, and entirely new segmentations reveal writing elsewhere, such as the outermost wrap of the scroll!

<div className="flex flex-wrap mb-4">
  <div className="w-[24%]"><img src="/img/grandprize/autoseg_paragraph_1.webp"/></div>
  <div className="w-[21%]"><img src="/img/grandprize/autoseg_paragraph_2.webp"/></div>
  <div className="w-[52%]"><img src="/img/grandprize/autoseg_paragraph_3_4.webp"/></div>
  <div className="w-[36.85%]"><img src="/img/grandprize/autoseg_outermost_sheet.webp"/></div>
  <div className="w-[30%]"><img src="/img/grandprize/autoseg_recto.webp"/></div>
  <div className="w-[30%]"><img src="/img/grandprize/autoseg_recto_64.webp"/></div>
  <figcaption className="mt-[-6px]">Outputs from auto-segmentation. The top row overlaps with the submission image, the bottom row has new segments. Much work remains to improve this promising tool.</figcaption>
</div>

Congratulations to Youssef, Luke, and Julian. You are the well-deserved winners of the 2023 Vesuvius Challenge Grand Prize!

## Runners up

Of the remaining submissions, the scores from our team of papyrologists identify a three-way tie for runner up. These entries show remarkably similar readability to each other, but still stand out from the rest by being significantly more readable. Congratulations to the following teams, each taking home \$50,000!

<div className="mb-4">
  <a target="_blank" href="/img/grandprize/sq_text_wbb.webp"><img src="/img/grandprize/sq_text_wbb-smaller.webp" className="w-[100%]"/></a>
  <figcaption className="mt-[-6px]">Shao-Qian Mah. <a href="https://github.com/SQMah/Vesuvius-Grand-Prize-Submission/">GitHub</a></figcaption>
</div>

<div className="mb-4">
  <a target="_blank" href="/img/grandprize/elian_text_wbb.webp"><img src="/img/grandprize/elian_text_wbb-smaller.webp" className="w-[100%]"/></a>
  <figcaption className="mt-[-6px]">Elian Rafael Dal Prá, Sean Johnson, Leonardo Scabini, Raí Fernando Dal Prá, João Vitor Brentigani Torezan, Daniel Baldin Franceschini, Bruno Pereira Kellm, Marcelo Soccol Gris, and Odemir Martinez Bruno. <a href="https://github.com/erdpx/vesuvius-grand-prize">GitHub</a></figcaption>
</div>

<div className="mb-4">
  <a target="_blank" href="/img/grandprize/lou_text_wbb.webp"><img src="/img/grandprize/lou_text_wbb-smaller.webp" className="w-[100%]"/></a>
  <figcaption className="mt-[-6px]">Louis Schlessinger and Arefeh Sherafati. <a href="https://github.com/lschlessinger1/vesuvius-grand-prize-submission">GitHub</a></figcaption>
</div>

These teams each brought to the table new approaches to the subtleties of ink labeling and sampling. Be sure to check out their methods at the links above. Other teams may also now choose to share their approaches, so be sure to follow our [Discord community](https://discord.com/invite/uTfNwwecCQ) for updates. Joining our community also provides access to the CT data and more images under our data agreement, as well as a front-row seat to daily discovery and collaboration!

## What does the scroll say?

To date, our efforts have managed to unwrap and read about 5% of the first scroll. Our eminent team of papyrologists has been hard at work and has achieved a preliminary transcription of all the revealed columns. We now know that this scroll is not a duplicate of an existing work; it contains never-before-seen text from antiquity. The papyrology team are preparing to deliver a comprehensive study as soon as they can. You all gave them a lot of work to do! Initial readings already provide glimpses into this philosophical text. From our scholars:

<div className="ml-8 mb-4">The general subject of the text is pleasure, which, properly understood, is the highest good in Epicurean philosophy. In these two snippets from two consecutive columns of the scroll, the author is concerned with whether and how the availability of goods, such as food, can affect the pleasure which they provide.</div>

<div className="ml-8 mb-4">Do things that are available in lesser quantities afford more pleasure than those available in abundance? Our author thinks not: <em>“as too in the case of food, we do not right away believe things that are scarce to be absolutely more pleasant than those which are abundant.”</em> However, is it easier for us naturally to do without things that are plentiful? <em>“Such questions will be considered frequently.”</em></div>

<div className="ml-8 mb-4">Since this is the end of a scroll, this phrasing may suggest that more is coming in subsequent books of the same work. At the beginning of the first text, a certain Xenophantus is mentioned, perhaps the same man — presumably a musician — also mentioned by Philodemus in his work <em>On Music</em>.</div>

[Philodemus](https://plato.stanford.edu/entries/philodemus/), of the Epicurean school, is thought to have been the philosopher-in-residence of the villa, working in the small library in which the scrolls were found.

Initial, rough draft transcriptions:

<div className="pl-8 flex w-[100%] flex-wrap">
  <div className="sm:w-[48%] sm:mr-4 max-w-[300px]">
    <a target="_blank" href="/img/grandprize/col-8.webp"><img className="w-[100%]" src="/img/grandprize/col-8.webp"/></a>
  </div>
  <div className="sm:w-[48%] text-[80%]">

```
    col. -8, ll. 2-14:
2   ...]ι̣μ̣εν τοὺϲ̣ [πα]ρ̣[ὰ Ξ]ε̣-
    νοφάντωι το̣ιούτου[ϲ,
    ὃ καὶ ὑπ’ ἄ̣λλων δοκεῖ
5   γείνεϲθαι, παραπλη-
    ϲίωϲ δ̣’ ο̣ὐδὲ παρ̣’ ἑτέρωι
    ἴδι̣ον το̣ῦ δ̣οκοῦ̣ντοϲ̣
    εἶναι καὶ παρὰ πλε̣ί-
    οϲ̣ι̣ν̣ ἥδιο̣ν, ἀλλ’ ὡ̣ϲ̣ καὶ
10  ἐ̣π̣ὶ τῶν βρω̣μ̣άτ̣ων
    ο̣ὐ̣κ ἤδ̣η τὰ ϲπάνια
    πάντωϲ̣ καὶ ἡδ̣ίω
    τῶν δ̣αψιλῶν̣ ε̣ἶναι̣
14  νομίζ̣ο̣με̣ν· οὐ γ̣ὰρ̣
```

  </div>
</div>

<div className="pl-8 flex w-[100%] flex-wrap mb-4">
  <div className="sm:w-[48%] sm:mr-4 max-w-[300px]">
    <a target="_blank" href="/img/grandprize/col-7.webp"><img className="w-[100%]" src="/img/grandprize/col-7.webp"/></a>
  </div>
  <div className="sm:w-[48%] text-[80%]">

```
    col. -7, ll. 4-10:
    λ̣ει παρὰ τὰ δαψιλῆ.
5   θεωρηθήϲεται δὲ τὰ
    τοιαῦθ’ οὕτω{ι} πολ̣λά-
    κιϲ πότερον ὅ̣ταν πα-
    ρῇ τὸ δαψιλέϲτερον
    ἡ φύϲιϲ ἥδιον ἀπαλλάτ-
10  τει το̣ύ̣τ̣ο̣υ̣ καὶ πάλ̣ι̣ν̣ ̣ ̣
```

  </div>
</div>

Later in the scroll:

<div className="ml-8 mb-4">In the closing section of the text our author takes a parting shot at his adversaries, who <em>“have nothing to say about pleasure, either in general or in particular, when it is a question of definition.”</em></div>

<div className="pl-8 flex w-[100%] flex-wrap mb-4">
  <div className="sm:w-[48%] sm:mr-4 max-w-[300px]">
    <a target="_blank" href="/img/grandprize/col-2.webp"><img className="w-[100%]" src="/img/grandprize/col-2.webp"/></a>
  </div>
  <div className="sm:w-[48%] text-[80%]">

```
    col. -2, ll. 2-8:
2   ἑ̣κάϲτηϲ κριτηρίων
    θεωροῦνται. πρὸϲ δὲ
    οὔτε καθόλου περὶ
5   ἡδονῆϲ ἐχόντων τι
    λέγειν οὔτε περὶ τῆϲ
    κατὰ μ̣έ̣ρο̣ϲ̣, ὅ̣τε ὡ-
8   ριϲμένον τι, ἀλλ’ οὖν
    …
```

  </div>
</div>

Finally the scroll concludes:

<div className="italic ml-8 mb-4"><em>“… for we do [not] refrain from questioning some things, but understanding/remembering others. And may it be evident to us to say true things, as they might have often appeared evident!”</em></div>

<div className="pl-8 flex w-[100%] flex-wrap">
  <div className="sm:w-[48%] sm:mr-4 max-w-[300px]">
    <a target="_blank" href="/img/grandprize/col-1.webp"><img className="w-[100%]" src="/img/grandprize/col-1.webp"/></a>
  </div>
  <div className="sm:w-[48%] text-[80%]">

```
    col. -1, ll. 1-6:
1   ὰρ ἀπ̣εχόμ̣ε̣θ̣α̣ τὰ
    μὲν κρίνειν, τὰ δὲ
    κατέχειν καὶ ἐμφαί
    νoιθ’ ἡμῖν ἀληθῆ λέ-
5   γειν ὥϲπερ πολλά̣κιϲ
    ἂν ἐ̣μφανε̣ίη̣{ι}.
```

  </div>
</div>

[Richard Janko](https://lsa.umich.edu/classics/people/departmental-faculty/rjanko.html) writes:

<div className="italic ml-8 mb-4">“Is the author Epicurus' follower, the philosopher and poet Philodemus, the teacher of Virgil? It seems very likely.</div>

<div className="italic ml-8 mb-4">Is he writing about the effect of music on the hearer, and comparing it to other pleasures like those of food and drink? Quite probably.</div>

<div className="italic ml-8 mb-4">Does this text come from his four-part treatise on music, of which we know Book 4? Quite possibly: the title should soon become available to read.</div>

<div className="italic ml-8 mb-4">Is the Xenophantus who is mentioned the celebrated flute-player, or the man famous in antiquity for being unable to control his laughter, or someone else entirely? So many questions! But improvements to the identification of the ink, which can be expected, will soon answer most of them. I can hardly wait.”</div>

[Federica Nicolardi](https://www.docenti.unina.it/federica.nicolardi) told us:

<div className="italic ml-8 mb-4">“Epicureanism says hi, with a text full of music, food, senses, and pleasure!”</div>

From [Bob Fowler](https://www.thebritishacademy.ac.uk/fellows/robert-fowler-FBA/):

<div className="italic ml-8 mb-4">“Like other Epicureans, he valued pleasure above all - but pleasure rightly understood, not mere indulgence. Living in ancient Rome - a society not known for abstinence - Philodemus could expect to meet with skepticism from his readers.”</div>

Scholars might call it a philosophical treatise. But it seems familiar to us, and we can’t escape the feeling that the first text we’ve uncovered is a 2000-year-old blog post about how to enjoy life. Is Philodemus throwing shade at the stoics in his closing paragraph, asserting that stoicism is an incomplete philosophy because it has “nothing to say about pleasure?” The questions he seems to discuss — life’s pleasures and what makes life worth living — are still on our minds today.

We can expect many more works from Philodemus in the current collection, once we’re able to scale up this technique. But there could be other text as well — an Aristotle dialog, a lost history of Livy, a lost Homeric epic work, a poem from Sappho — who knows what treasures are hidden in these lumps of ash.

And there is the hope of a much bigger library still in the ground, since two levels of the villa remain unexcavated. More about this below!


## How accurate are these pictures?

Machine learning models are infamous for “hallucinating”: making up text or pictures that look similar to their training data. Similarly, there might be ways for contestants to cheat by making up images themselves, e.g. by embedding those in the model weights. How do we know that that’s not happening here? There are a couple of answers:

* **Technical reproduction.** The Vesuvius Challenge Technical Review Team reproduced the winning submissions manually. We made sure to clearly understand every part of the code, and that when we run it independently we get similar output images. Since all code and training data is now open source, you can do the same!
* **Multiple submissions of the same area.** You might have noticed that all submission images above show the same area of the scroll. This is because we released 3d-mapped papyrus sheets within the CT-scan (“segments”) created by our segmentation team, which were then used by all contestants. The resulting output images — created by different ML models and training labels — have produced extremely similar results. This holds not just for the winners and runner ups, but also for the other submissions that we received.
* **Small input/output windows.** The ink detection models are not based on Greek letters, optical character recognition (OCR), or language models. Instead, they independently detect tiny spots of ink in the CT scan, the writing appearing later when these are aggregated. As a result, the text appearing in the images is not the imagined output of a machine learning model, but is instead directly tied to the underlying data in the CT scan.

<figure>
  <video autoPlay playsInline loop muted className="w-[100%]" poster="/img/tutorials/ink-detection-anim3-dark.webp">
    <source src="/img/tutorials/ink-detection-anim3-dark.webm" type="video/webm"/>
  </video>
  <figcaption className="mt-[-6px]">The models use small input/output windows. In some cases, the output is even only binary (“ink” vs “no ink”), as shown in this animation. This makes it extremely unlikely for the model to hallucinate shapes that look like letters.</figcaption>
</figure>

## How does the unwrapping work?

<div>Roughly, virtual unwrapping works in <a href="tutorial">three steps:</a></div>

1. **Scanning:** creating a 3D scan of a scroll or fragment using X-ray tomography.
2. **Segmentation:** tracing the crumpled layers of the rolled papyrus in the 3D scan and then unwrapping, or flattening, them.
3. **Ink Detection:** identifying the inked regions in the flattened segments using a machine learning model.

These scrolls were scanned at Diamond Light Source, a particle accelerator near Oxford, England. The facility produces a parallel beam of X-rays at high flux, allowing for fast, accurate, and high-resolution imaging. The X-ray photos are turned into a 3D volume of voxels using tomographic reconstruction algorithms, resulting in a stack of slice images.

<figure>
  <video autoPlay playsInline loop muted className="w-[100%] max-w-[600px]" poster="/img/grandprize/.webp">
    <source src="/img/grandprize/scroll1.webm" type="video/webm"/>
  </video>
  <figcaption className="mt-[-6px]">Scrubbing through the slice images of the scroll.</figcaption>
</figure>

The next step is to identify individual sheets of papyrus in 3D space. For this we primarily use a tool called [Volume Cartographer](https://github.com/educelab/volume-cartographer), created by Seth Parker and others in Brent Seales’ lab, and augmented by our contestants, primarily Julian Schilliger (Grand Prize winner) and Philip Allgaier.

Volume Cartographer is operated by our team of full-time segmenters: Ben Kyles, David Josey, and Konrad Rosenberg. They use a combination of automatic algorithms and manual adjustments to map out large areas of papyrus. This is still a painstaking process, with lots of room for improvement if we’re going to segment all the scrolls.

<figure>
  <video autoPlay playsInline loop muted className="max-w-[100%] rounded-xl" poster="/img/tutorials/vc-extrapolation2.webp">
    <source src="/img/tutorials/vc-extrapolation2.webm" type="video/webm"/>
  </video>
  <figcaption className="mt-[-6px]">Animation showing manual and automatic segmentation in Volume Cartographer.</figcaption>
</figure>

Finally, ink detection. Stephen Parsons at Brent’s lab had [shown](https://uknowledge.uky.edu/cs_etds/138/) that Herculaneum ink could theoretically be detected in CT scans, but so far only using smaller fragments — detecting ink in the larger scans of complete scrolls had yet to be achieved. For months this part proved elusive, until progress was made on two separate tracks:

1. **Crackle pattern.** Last summer, Casey Handmer <a href="https://caseyhandmer.wordpress.com/2023/08/05/reading-ancient-scrolls/">discovered</a> a strange pattern of “crackle” by looking at raw flattened surface volumes. This pattern appeared to form letters. Casey won the First Ink Prize for this monumental discovery and shared it with the community, and a flurry of activity followed.

<figure className="max-w-[500px] ml-8">
    <img className="w-[47%] mr-4" src="/img/firstletters/pi1.webp" />
    <img className="w-[47%]" src="/img/firstletters/pi2.webp" />
</figure>

<div className="ml-8 mb-4">Luke Farritor (Grand Prize winner), immediately started hunting for more crackle in flattened surface volumes produced by the segmentation team. He then trained a machine learning model on the shapes he found, which led directly to him winning the <a href="/firstletters">First Letters Prize</a> in October.</div>

2. **Kaggle competition.** Separately, <a href="https://www.kaggle.com/competitions/vesuvius-challenge-ink-detection">hundreds of teams</a> tried building the best machine learning model for detecting ink in open fragments — pieces that had broken off during the physical unrolling process of scrolls, hundreds of years ago. Instead of labeling crackle (which wasn’t known yet), they had the benefit of ground truth data directly from photos of these fragments.

<div className="flex flex-wrap ml-8">
  <div className="sm:w-[32%] mb-2 mr-2" style={{ maxWidth: "calc(33% - 8px)" }}><img src="/img/data/fr1.webp" className="w-[100%]"/><figcaption className="mt-[-6px]">Photo of Fragment 1</figcaption></div>
  <div className="sm:w-[30%] mb-2 mr-2" style={{ maxWidth: "calc(33% - 8px)" }}><img src="/img/data/ir-fr1.webp" className="w-[100%]"/><figcaption className="mt-[-6px]">Aligned infrared</figcaption></div>
  <div className="sm:w-[30%] mb-2 mr-2" style={{ maxWidth: "calc(33% - 8px)" }}><img src="/img/data/inklabels-fr1.webp" className="w-[100%]"/><figcaption className="mt-[-6px]">Aligned binary ink labels</figcaption></div>
</div>

<div className="ml-8 mb-4">This resulted in excellent models, but they did not seem to work on the flattened segments which the segmentation team produced. That was, until Youssef Nader (Grand Prize winner) used domain adaptation techniques on them, the start of a technique that ultimately won him the second place First Letters Prize.</div>

After the success of the First Letters Prize, the Grand Prize seemed within reach. Youssef, Luke, and Julian teamed up, with several other teams putting in strong submissions as well.

## What did it take?

With Vesuvius Challenge, we hope not only to solve the problem of reading the Herculaneum Papyri, but also to inspire similar projects. For that, it’s helpful to know what has contributed to our success in 2023. Here are some things we believe were important:

1. **An inspiring goal and a clear target.** There are many worthy causes in the world, so it helps that our goal is unusual for a computing competition. It drew more press and donations early on, it attracted an intrinsically motivated community, and it increased our probability of success to begin with (emerging research area => a higher marginal utility of dollars spent). We’d love to see more projects that are “out there,” for exactly these reasons!

<div className="mb-4 ml-8">
  <img src="/img/grandprize/banner.webp" className="w-[100%] max-w-[400px]"/>
</div>

2. **A solid starting point.** The foundation was laid by <a href="https://www2.cs.uky.edu/dri/">Dr. Seales and his team</a>. They spent two decades making the first scroll scans, building <a href="community_projects#volume-cartographer">Volume Cartographer</a>, demonstrating the <a href="https://www2.cs.uky.edu/dri/the-scroll-from-en-gedi/">first success</a> in virtual unwrapping, and <a href="https://uknowledge.uky.edu/cs_etds/138/">proving</a> that Herculaneum ink can be detected in CT.

<div className="mb-4 ml-8">
  <img src="/img/landing/brent1.webp" className="w-[100%] max-w-[600px]"/>
  <figcaption className="mt-[-6px]">Brent Seales, Seth Parker, and Michael Drakopoulos at the particle accelerator.</figcaption>
</div>

3. **Blending competition and cooperation.** A Grand Prize on its own would suffer from information “hoarding”: no one would share their intermediate work, because others could take it and use it to beat them to the finish line. Without information sharing, the probability of a single team solving all the puzzle pieces to win the Grand Prize would be dramatically lower.

<div className="ml-8 mb-4">Instead, we blended competition and cooperation by adding <a href="/winners">“progress prizes”</a> along the way. These were smaller prizes (often in the \$1,000-10,000 range) every ~2 months. To win a progress prize, you had to publish your code or research as open source, thereby benefiting the entire community.</div>

<div className="mb-4 ml-8">
  <img src="/img/grandprize/winners1.webp" className="w-[100%]"/>
  <img src="/img/grandprize/winners3.webp" className="w-[100%]"/>
  <figcaption className="mt-[-6px]">Some of the many prize winners.</figcaption>
</div>

<div className="ml-8 mb-4">Besides “leveling up” the entire community, this had several side benefits. We generated buzz and excitement in the community, which was motivating for everyone. It allowed winners to re-invest their winnings into better equipment, compute time, or even reduced hours at work or study to dedicate more to the competition. And it allowed people to find each other and form teams — like we saw with the Grand Prize winners.</div>

4. **Hiring an in-house segmentation team.** Every week we asked ourselves: what is the best thing we can do now to maximize the chance that someone wins the Grand Prize? In early summer, this led to the (then somewhat controversial) decision of hiring a full-time team of data labelers to manually trace the papyrus inside the scrolls and open source the flattened segments.

<div className="ml-8 mb-4">An alternative was to leave the problem of segmentation to the contestants, or even to award separate prizes for segments, but this had several downsides. First, it’s hard to judge segment quality before knowing what to look for (we didn’t have working ink detection yet). Incentivizing segment quantity would automatically penalize quality. Second, labeling work is tedious and time consuming, and turned out to have a long learning curve, so it’s desirable to guarantee some compensation, which can’t be done with a prize. Third, the feedback loop with prizes can be pretty long.</div>

<div className="ml-8 mb-4">We were not dogmatically attached to just being referees; we were willing to run out onto the field and kick the ball a little. So we did what we thought would maximize success, and for the critical bottleneck of segmentation, that meant hiring a team.</div>

<div className="flex flex-wrap mb-4 ml-8">
  <div className="w-[30%] mr-4 mb-2"><img src="/img/grandprize/ben-smaller.webp"/></div>
  <div className="w-[30%] mr-4 mb-2"><img src="/img/grandprize/david.webp"/></div>
  <div className="w-[25%]"><img src="/img/grandprize/konrad-smaller.webp"/></div>
  <figcaption className="mt-[-6px]">Our wonderful segmentation team: Ben, David, and Konrad. Also a big shout-out to former team members!</figcaption>
</div>

<div className="ml-8 mb-4">Ultimately, this decision worked out great. It led directly to Casey Handmer’s <a href="https://caseyhandmer.wordpress.com/2023/08/05/reading-ancient-scrolls/">discovery</a> of the “crackle pattern” — the first directly visible evidence of ink and letters within the complete scrolls. The in-house segmentation expertise also turned out to be invaluable throughout the rest of the competition, discovering areas with potentially high ink signal, and figuring out the intricacies of the structure of the scrolls. And the segmenters worked closely with community contestants, which led to much better segmentation software. It was the best of both worlds!</div>

5. **Maximizing surface area for breakthroughs.** Our success was the result of many smaller breakthroughs by a broad group of people. It’s remarkable how many things had to come together to make this happen. Remove any of these, and we would not have succeeded, at least not within this timeframe.

<div className="mb-4 ml-8">
  <img src="/img/grandprize/steps.webp" className="w-[100%] max-w-[400px]"/>
</div>

<div className="ml-8 mb-4">There are many more contributions than we can list here - even ideas and discoveries landing outside the critical path were still important, because a massive search space had to be exhausted to find the ideas that worked. Given the right framework, the collective intelligence of a community like this is very powerful.</div>

## What’s next? Announcing the 2024 Vesuvius Challenge Grand Prize.

When we started the competition, most of us estimated that we had a less than 30% probability of success within the year. At that point, no letters had yet been discovered inside of a scroll. On top of that, the scrolls had barely been segmented at all. We had doubt as to whether the project could succeed, especially by the deadline we set. Was the ink signal present? Were the scans high-resolution enough? Could the techniques used to identify ink in the fragments be transferred into a scroll? None of this was known at the time. But we knew it was worth trying!

In Stage 1 of Vesuvius Challenge, we answered all of these questions, extracting 15 columns of never-before-seen text from inside a lump of carbon. We now have proven techniques for virtually unwrapping the papyrus scroll and recognizing ink using machine learning.

### Vesuvius Challenge Stage 2

In 2023 we got from 0% to 5% of a scroll. In 2024 our goal is to go from 5% of one scroll, to 90% of all four scrolls we have scanned, and to lay the foundation to read all 300  scrolls.

The primary goal for 2024 is to read 90% of the scrolls, and **we will issue the 2024 Grand Prize to the first team that is able to do this**. More details on the exact grand prize judging criteria will be available in March.

The bottleneck to achieve this milestone is the process of tracing the surface of the papyrus inside the scroll. Today this is extremely manual. It cost us more than \$100 per square centimeter in manual labor to produce the text we can read today. At this price, it would cost hundreds of millions or maybe even billions of dollars to segment all of the scrolls. While improvements to our segmentation tools have increased our efficiency, it is still far too manual and expensive. What we need is automation.

And so our primary goal for stage 2 is to perfect autosegmentation. Done right, this will also allow us to read the most challenging regions within the scroll – areas where the scroll was heavily compressed, cracked, delaminated, or otherwise damaged – which in many cases our current tools cannot even penetrate.

In 2023 we were amazed by the community contributions. We loved the competition for the grand prize, which brought out the best in the contestants, but we were also thrilled to see the community collaborate towards intermediate goals. In 2024 we are leaning into that, still offering a grand prize, but allocating even more of the prize pool towards community contributions – a pool that will grow as we raise more money.

We’re also planning to help speed things along ourselves, balancing prizes and in-house expertise to continue the collaboration that worked so well in 2023. To this end, we’ll hire a small software/ML team, in addition to the full-time segmentation team, who will work in the open with our community to advance the state of the art.

If you are interested in contributing to our funding, and joining the craziest archeological project in existence, please contact us.

### And after that?

After that, we will scan and read every scroll. We estimate that the scrolls we have in Naples contain more than 16 megabytes of text. Some members of our papyrology team say that revealing this text will be the greatest revolution in the classics since the Renaissance. However it goes, it will certainly be fun to try!

And as if the prospect of reading hundreds of scrolls isn’t good enough, there might be an even bigger payoff at the end of all of this (as Nat said on the [Dwarkesh podcast](https://youtu.be/qcvMjoJdck4?t=646): <em>“there is gold in this mud”</em>).

<div>Dr. Garrett Ryan (<a href="https://www.youtube.com/@toldinstone">toldinstone</a>) writes it best on our <a href="background">History page</a>:</div>

<div className="italic ml-8 mb-4">“The scrolls we have now may be just the beginning. When part of the Villa of the Papyri was cleared in the 1990s, archaeologists realized that the building was much larger than previously thought, with two unexcavated levels. At the very least, these floors likely contain more papyri in cabinets and carrying cases. And it’s probable that they conceal a far greater treasure.</div>

<div className="italic ml-8 mb-4">We have not yet found the villa’s main library, which would have contained a much wider range of Greek and Latin literature. That library, with its thousands or even tens of thousands of scrolls, must still be buried. If those texts are discovered, and if even a small fraction can still be read, they will transform our knowledge of classical life and literature on a scale not seen since the Renaissance.”</div>

The potential of tens of thousands of scrolls, still buried, waiting to be discovered?! The most exciting days [still lay ahead](https://twitter.com/natfriedman/status/1712123310551548222).

Read more detail about what comes next in our [Master Plan](master_plan).

<div className="mb-4">
  <img src="/img/landing/rocio-espin-pinar-villa-papyri-small.webp" className="w-[100%] max-w-[600px]"/>
</div>

## Thank you

<div>This couldn’t have happened without the many, many people who contributed in various ways, and we’d like to say thanks to all of them:</div>

* Everyone who competed, shared insights, wrote code, made analyses, and brought energy to the project.
* Our adventurous donors, all of whom are private individuals from the tech world, who supported this project when it was not at all clear that there was any chance of success.
* The organizing teams (listed on our homepage): Vesuvius Challenge team, EduceLab team, and Papyrology team.
* Our partners: EduceLab, Institut de France, Diamond Light Source, Biblioteca Nazionale di Napoli, the Getty, and Kaggle — and all their respective funders.
* The professional and amateur papyrologists, historians, classicists, and other scholars — who helped answer countless questions in Discord.
* The supporting staff on the Vesuvius Challenge side (Sean, Emily, Frank, Lulu), and on the University of Kentucky side (Lindsey, Eric).
* The many contributors to the cause who came before us — who did excavations, wrote code, made scans, and built machines out of catgut and pig bladders to try to physically unroll the scrolls.
* And of course the Grand Prize winners!


Thank you all so much!!

Now let’s get on with it and read the rest of the scrolls. The best is yet to come.

*Join us for a celebration at the Getty Villa Museum in Los Angeles on March 16th, 4pm. <a href="https://www.getty.edu/visit/cal/events/ev_4074.html">More information here</a>.*
