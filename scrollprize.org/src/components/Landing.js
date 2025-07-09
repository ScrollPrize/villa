import React, { useEffect, useRef, useState } from "react";
import useBrokenLinks from "@docusaurus/useBrokenLinks";
import BeforeAfter from "./BeforeAfter";
import LatestPosts from "./LatestPosts";

const inlineImage = (src) => (
  <div
    className="md:mb-8 mb-4 rounded-lg md:h-80 h-full md:w-auto w-full aspect-[4/3] sepia-[.4] inline-block mr-4"
    style={{ backgroundImage: `url(${src})`, backgroundSize: "cover" }}
  />
);

const stories = ({ unrollVideo, mlVideo, xrayVideo }) => [
  {
    date: "79 AD",
    text: "Mount Vesuvius erupts.",
    anchor: "vesuvius",
    description: (
      <>
        <div className="mb-4 max-w-3xl">
          In Herculaneum, twenty meters of hot mud and ash bury an enormous
          villa once owned by the father-in-law of Julius Caesar. Inside, there
          is a vast library of papyrus scrolls.
        </div>
        <div className="mb-8 max-w-3xl">
          The scrolls are carbonized by the heat of the volcanic debris. But
          they are also preserved. For centuries, as virtually every ancient
          text exposed to the air decays and disappears, the library of the
          Villa of the Papyri waits underground, intact.
        </div>
        {inlineImage("/img/landing/rocio-espin-pinar-villa-papyri-small.webp")}
      </>
    ),
    background: "/img/landing/story1.webp",
  },
  {
    date: "1750 AD",
    text: "A farmer discovers the buried villa.",
    description: (
      <>
        <div className="max-w-3xl mb-8">
          While digging a well, an Italian farmworker encounters a marble
          pavement. Excavations unearth beautiful statues and frescoes – and
          hundreds of scrolls. Carbonized and ashen, they are extremely fragile.
          But the temptation to open them is great; if read, they would
          significantly increase the corpus of literature we have from
          antiquity.
        </div>
        <div className="max-w-3xl mb-8">
          Early attempts to open the scrolls unfortunately destroy many of them.
          A few are painstakingly unrolled by a monk over several decades, and
          they are found to contain philosophical texts written in Greek. More
          than six hundred remain unopened and unreadable.
        </div>
        {inlineImage("/img/landing/scroll.webp")}
        {inlineImage("/img/landing/herc-materials.webp")}
      </>
    ),
    background: "/img/landing/story2.webp",
  },
  {
    date: "2015 AD",
    text: "Dr. Brent Seales pioneers virtual unwrapping.",
    description: (
      <>
        <div className="max-w-3xl mb-4">
          Using X-ray tomography and computer vision, a team led by Dr. Brent
          Seales at the University of Kentucky reads the En-Gedi scroll without
          opening it. Discovered in the Dead Sea region of Israel, the scroll is
          found to contain text from the book of Leviticus.
        </div>
        <div className="max-w-3xl mb-8">
          Virtual unwrapping has since emerged as a growing field with multiple
          successes. Their work went on to show the elusive carbon ink of the
          Herculaneum scrolls can also be detected using X-ray tomography,
          laying the foundation for Vesuvius Challenge.
        </div>
        <video
          // autoPlay
          playsInline
          loop
          muted
          className="md:mb-8 mb-4 rounded-lg md:h-80 h-full md:w-auto w-full aspect-[4/3] sepia-[.8] inline-block mr-4 object-cover"
          poster="/img/landing/engedi5.webp"
          ref={unrollVideo}
        >
          <source src="/img/landing/engedi5.webm" type="video/webm" />
        </video>
        {inlineImage("/img/landing/brent1.webp")}
      </>
    ),
    background: "/img/landing/story3.webp",
  },
  {
    date: "2023 AD",
    text: "A remarkable breakthrough.",
    description: (
      <>
        <div className="max-w-3xl mb-8">
          Vesuvius Challenge was launched in March 2023 to bring the world
          together to read the Herculaneum scrolls. Along with smaller progress
          prizes, a Grand Prize was issued for the first team to recover 4
          passages of 140 characters from a Herculaneum scroll.
        </div>
        <div className="max-w-3xl mb-8">
          Following a year of remarkable progress,{" "}
          <a href="grandprize">the prize was claimed</a>. After 275 years, the
          ancient puzzle of the Herculaneum Papyri has been cracked open. But
          the quest to uncover the secrets of the scrolls is just beginning.
        </div>
        <div className="flex overflow-hidden rounded-lg md:mb-8 mb-4 h-96 relative bg-black">
          <img
            src="/img/landing/scroll-full-min.webp"
            className="pan-horizontal max-w-none"
          />
        </div>
      </>
    ),
    background: "/img/landing/story5.webp",
  },
  {
    date: "2024 AD",
    text: "New frontiers.",
    description: (
      <>
        <div className="max-w-3xl mb-8">
          <p>
            A widespread community effort builds on the success of the first scroll,
            automating and refining the components of the virtual unwrapping pipeline.
            Efforts to scan and read multiple scrolls are underway.
            New text is revealed from another scroll.
          </p>
          <div className="flex flex-col sm:flex-row gap-4 mt-4">
            <img
              src="/img/landing/patches.webp"
              alt="Community Effort 1"
              className="w-full sm:w-1/2 object-cover rounded-lg"
            />
            <img
              src="/img/landing/scroll5.webp"
              alt="Community Effort 2"
              className="w-full sm:w-1/2 object-cover rounded-lg"
            />
          </div>
        </div>
      </>
    ),
    background: "/img/landing/story6.webp",
  },
];

const prizes = [
  {
    title: "First Title Prize",
    prizeMoney: "$60,000",
    description: "Discover the end title of a sealed Herculaneum scroll.",
    requirement: "",
    href: "https://scrollprize.substack.com/p/60000-first-title-prize-awarded",
    winners: [
      {
        name: "Marcel Roth",
        image: "/img/landing/marcel.webp",
      },
      {
        name: "Micha Nowak",
        image: "/img/landing/micha.webp",
      },
    ],
    won: true,
    bannerImage: "/img/landing/scroll5-title-boxes.webp",
  },
  {
    title: "First Automated Segmentation Prize",
    prizeMoney: "$60,000",
    description: "Reproduce the 2023 Grand Prize result but faster",
    requirement: "",
    href: "https://scrollprize.substack.com/p/awarding-the-amazing-autosegmentation",
    winners: [
      // {
      //   name: "Paul Henderson",
      //   image: "/img/landing/paul.webp",
      // },
      // {
      //   name: "Hendrik Schilling",
      //   image: "/img/landing/hendrik.webp",
      // },
      {
        name: "Sean Johnson",
        image: "/img/landing/sean.webp",
      },
    ],
    winnersLabel: "3 Winners",
    won: true,
    bannerImage: "/img/landing/patches.webp",
  },
  {
    title: "2023 Grand Prize",
    prizeMoney: "$850,000",
    description: "First team to read a scroll by December 31st 2023",
    requirement: "",
    winnersLabel: "4 Winning Teams",
    winners: [
      {
        name: "Youssef Nader",
        image: "/img/landing/youssef.webp",
      },
      {
        name: "Luke Farritor",
        image: "/img/landing/luke.webp",
      },
      {
        name: "Julian Schilliger",
        image: "/img/landing/julian.webp",
      },
    ],
    bannerImage: "/img/landing/grand-prize-preview.webp",
    href: "/grandprize",
  },
  {
    title: "First Letters & First Ink",
    prizeMoney: "$60,000",
    description: "Detect 10 letters in a 4 cm² area in a scroll",
    requirement: "",
    winners: [
      {
        name: "Luke Farritor",
        image: "/img/landing/luke.webp",
      },
      {
        name: "Youssef Nader",
        image: "/img/landing/youssef.webp",
      },
      {
        name: "Casey Handmer",
        image: "/img/landing/casey.webp",
      },
    ],
    // bannerImage: "/img/landing/first-letters.webp",
    href: "/firstletters",
  },
  {
    title: "Open Source Prizes",
    prizeMoney: "$200,000+",
    description: "",
    requirement: "",
    winnersLabel: "50+ Winners",
    winners: [
      {
        name: "Giorgio Angelotti",
        image: "/img/landing/giorgio.webp",
      },
      {
        name: "Yao Hsiao",
        image: "/img/landing/yao.webp",
      },
      {
        name: "Brett Olsen",
        image: "/img/landing/brett.webp",
      },
    ],
    won: true,
    href: "/winners",
  },
  {
    title: "Ink Detection Prizes",
    prizeMoney: "$112,000",
    description: "Detect 10 letters in a 4 cm² area in a scroll",
    requirement: "",
    winnersLabel: "16 Winners",
    winners: [
      {
        name: "Yannick Kirchoff",
        image: "/img/landing/yannick.webp",
      },
      {
        name: "tattaka",
        image: "/img/landing/tattaka.webp",
      },
      {
        name: "Ryan Chesler",
        image: "/img/landing/ryan.webp",
      },
      {
        name: "Felix Yu",
        image: "/img/landing/felix.webp",
      },
    ],
    href: "/winners",
  },
  // {
  //   title: "Segmentation Prizes",
  //   prizeMoney: "$90,000",
  //   description: "Detect 10 letters in a 4 cm² area in a scroll",
  //   requirement: "",
  //   winnersLabel: "12 Winners",
  //   winners: [
  //     {
  //       name: "Ahron Wayne",
  //       image: "/img/landing/ahron.webp",
  //     },
  //     {
  //       name: "Julian Schilliger",
  //       image: "/img/landing/julian.webp",
  //     },
  //     {
  //       name: "Santiago Pelufo",
  //       image: "/img/landing/santiago.webp",
  //     },
  //     {
  //       name: "Yao Hsiao",
  //       image: "/img/landing/yao.webp",
  //     },
  //   ],
  //   won: true,
  //   href: "/winners",
  // },
  {
    title: "Read Entire Scroll Prize",
    prizeMoney: "$200,000",
    description: "Read an entire scroll",
    requirement: "",
    href: "prizes#read-entire-scroll-prize-200000",
  },
  {
    title: "First Letters / First Title Prizes",
    prizeMoney: "7 x $60,000",
    description:
      "Find the first letters or the title of a scroll",
    requirement: "",
    href: "prizes#first-letters-and-title-prizes",
  },
  {
    title: "Monthly Progress Prizes",
    prizeMoney: "$350,000",
    description: "Open ended prizes from $1,000-20,000",
    requirement: "",
    href: "prizes#progress-prizes",
  },
];

const creators = [
  {
    name: "Nat Friedman",
    image: "/img/landing/nat.webp",
    href: "https://nat.org/",
  },
  {
    name: "Daniel Gross",
    image: "/img/landing/daniel.webp",
    href: "https://dcgross.com/",
  },
  {
    name: "Dr. Brent Seales",
    image: "/img/landing/brent.webp",
    href: "https://educelab.engr.uky.edu/w-brent-seales",
  },
];

const sponsors = [
  {
    name: "Musk Foundation",
    amount: 2084000,
    href: "https://www.muskfoundation.org/",
    image: "/img/landing/musk.webp",
  },
  {
    name: "Alex Gerko",
    amount: 450000,
    href: "https://www.xtxmarkets.com/",
    image: "/img/landing/gerko.webp",
  },
  {
    name: "Joseph Jacks",
    amount: 250000,
    href: "https://twitter.com/JosephJacks_",
    image: "/img/landing/Joseph Jacks.webp",
  },
  {
    name: "Nat Friedman",
    amount: 225000,
    href: "https://nat.org/",
    image: "/img/landing/nat.webp",
  },
  {
    name: "Daniel Gross",
    amount: 225000,
    href: "https://dcgross.com/",
    image: "/img/landing/daniel.webp",
  },
  {
    name: "Matt Mullenweg",
    amount: 150000,
    href: "https://ma.tt/",
    image: "/img/landing/Matt Mullenweg.webp",
  },
  {
    name: "Emergent Ventures",
    amount: 100000,
    href: "https://www.mercatus.org/emergent-ventures",
  },
  {
    name: "Matt Huang",
    amount: 50000,
    href: "https://twitter.com/matthuang",
    image: "/img/landing/Matt Huang.webp",
  },
  {
    name: "John & Patrick Collison",
    amount: 125000,
    href: "https://stripe.com/",
    image: ["/img/landing/collison1.webp", "/img/landing/collison2.webp"],
  },
  {
    name: "Julia DeWahl & Dan Romero",
    amount: 100000,
    href: "https://twitter.com/natfriedman/status/1637959778558439425",
    image: [
      "/img/landing/Julia DeWahl.webp",
      "/img/landing/Dan Romero.webp",
    ],
  },
  {
    name: "Eugene Jhong",
    amount: 100000,
    href: "https://twitter.com/ejhong",
    image: "/img/landing/Eugene Jhong.webp",
  },
  {
    name: "Anonymous",
    amount: 100000,
    href: "https://www.youtube.com/watch?v=JqrJ4wGid4Y",
    image: "/img/landing/mystery.webp",
  },
  {
    name: "Bastian Lehmann",
    amount: 75000,
    href: "https://twitter.com/Basti",
    image: "/img/landing/Bastian Lehmann.webp",
  },
  {
    name: "Tobi Lutke",
    amount: 75000,
    href: "https://twitter.com/tobi",
    image: "/img/landing/Tobi Lutke.webp",
  },
  {
    name: "Guillermo Rauch",
    amount: 50000,
    href: "https://rauchg.com/",
    image: "/img/landing/Guillermo Rauch.webp",
  },
  {
    name: "Arthur Breitman",
    amount: 50000,
    href: "https://ex.rs/",
    image: "/img/landing/Arthur Breitman.webp",
  },
  {
    name: "Anonymous",
    amount: 50000,
    href: "https://www.youtube.com/watch?v=JqrJ4wGid4Y",
    image: "/img/landing/mystery.webp",
  },
  {
    name: "Anonymous",
    amount: 50000,
    href: "https://www.youtube.com/watch?v=JqrJ4wGid4Y",
    image: "/img/landing/mystery.webp",
  },
  {
    name: "Aaron Levie",
    amount: 25000,
    href: "https://twitter.com/levie",
    image: "/img/landing/Aaron Levie.webp",
  },
  {
    name: "Akshay Kothari",
    amount: 25000,
    href: "https://twitter.com/akothari",
    image: "/img/landing/Akshay Kothari.webp",
  },
  {
    name: "Alexa McLain",
    amount: 25000,
    href: "https://twitter.com/alexamclain",
    image: "/img/landing/Alexa McLain.webp",
  },
  {
    name: "Anjney Midha",
    amount: 25000,
    href: "https://twitter.com/AnjneyMidha",
    image: "/img/landing/Anjney Midha.webp",
  },
  {
    name: "franciscosan.org",
    amount: 25000,
    href: "https://franciscosan.org/",
    image: "/img/landing/franciscosan.webp",
  },
  {
    name: "John O'Brien",
    amount: 25000,
    href: "https://twitter.com/jobriensf",
    image: "/img/landing/John O'Brien.webp",
  },
  {
    name: "Mark Cummins",
    amount: 25000,
    href: "https://twitter.com/mark_cummins",
    image: "/img/landing/Mark Cummins.webp",
  },
  {
    name: "Jamie Cox & Gary Wu",
    amount: 15000,
    href: "https://www.fluidstack.io/",
    image: ["/img/landing/Jamie Cox.webp", "/img/landing/Gary Wu.webp"],
  },
  {
    name: "Mike Mignano",
    amount: 15000,
    href: "https://mignano.co/",
    image: "/img/landing/Mike Mignano.webp",
  },
  {
    name: "Aravind Srinivas",
    amount: 10000,
    href: "https://twitter.com/AravSrinivas",
    image: "/img/landing/Aravind Srinivas.webp",
  },
  {
    name: "Brandon Reeves",
    amount: 10000,
    href: "https://www.luxcapital.com/people/brandon-reeves",
    image: "/img/landing/Brandon Reeves.webp",
  },
  {
    name: "Brandon Silverman",
    amount: 10000,
    href: "https://twitter.com/brandonsilverm",
    image: "/img/landing/Brandon Silverman.webp",
  },
  {
    name: "Chet Corcos",
    amount: 10000,
    href: "https://chetcorcos.com",
    image: "/img/landing/Chet Corcos.webp",
  },
  {
    name: "Ivan Zhao",
    amount: 10000,
    href: "https://twitter.com/ivanhzhao",
    image: "/img/landing/Ivan Zhao.webp",
  },
  {
    name: "Neil Parikh",
    amount: 10000,
    href: "https://www.neilparikh.com/",
    image: "/img/landing/Neil Parikh.webp",
  },
  {
    name: "Stephanie Sher",
    amount: 10000,
    href: "https://twitter.com/stephxsher",
    image: "/img/landing/Stephanie Sher.webp",
  },
  {
    name: "Raymond Russell",
    amount: 10000,
    href: "https://twitter.com/raymondopolis",
    image: "/img/landing/Raymond Russell.webp",
  },
  {
    name: "Vignan Velivela",
    amount: 10000,
    href: "https://vignanv.com/",
    image: "/img/landing/Vignan Velivela.webp",
  },
  {
    name: "Katsuya Noguchi",
    amount: 10000,
    href: "https://twitter.com/kn",
    image: "/img/landing/Katsuya Noguchi.webp",
  },
  {
    name: "Shariq Hashme",
    amount: 10000,
    href: "https://shar.iq/",
    image: "/img/landing/Shariq Hashme.webp",
  },
  {
    name: "Sahil Chaudhary",
    amount: 10000,
    href: "https://twitter.com/csahil28",
    image: "/img/landing/Sahil Chaudhary.webp",
  },
  {
    name: "Maya & Taylor Blau",
    amount: 10000,
    href: "https://ttaylorr.com/",
    image: ["/img/landing/Maya Blau.webp", "/img/landing/Taylor Blau.webp"],
  },
  {
    name: "Matias Nisenson",
    amount: 10000,
    href: "https://twitter.com/MatiasNisenson",
    image: "/img/landing/Matias Nisenson.webp",
  },
  {
    name: "Mikhail Parakhin",
    amount: 10000,
    href: "https://twitter.com/mparakhin",
    image: "/img/landing/Mikhail Parakhin.webp",
  },
  {
    name: "Alex Petkas",
    amount: 5000,
    href: "https://twitter.com/costofglory",
    image: "/img/landing/Alex Petkas.webp",
  },
  {
    name: "Amjad Masad",
    amount: 5000,
    href: "https://twitter.com/amasad",
    image: "/img/landing/Amjad Masad.webp",
  },
  {
    name: "Conor White-Sullivan",
    amount: 5000,
    href: "https://twitter.com/Conaw",
    image: "/img/landing/Conor White-Sullivan.webp",
  },
  {
    name: "Will Fitzgerald",
    amount: 5000,
    href: "https://github.com/willf",
    image: "/img/landing/Will Fitzgerald.webp",
  },
];

const team = {
  challenge: [
    {
      name: "Nat Friedman",
      title: "Instigator & Founding Sponsor",
      href: "https://nat.org/",
    },
    {
      name: "Daniel Gross",
      title: "Founding Sponsor",
      href: "https://dcgross.com/",
    },
    {
      name: "Giorgio Angelotti",
      title: "Research Project Lead, PhD",
      href: "https://thegiorgio.org/",
    },
    {
      name: "Stephen Parsons",
      title: "Staff Software Engineer, PhD",
      href: "https://www2.cs.uky.edu/dri/stephen-parsons/",
    },
    {
      name: "Sean Johnson",
      title: "Technical Community Manager",
      href: "https://github.com/bruniss",
    },
    {
      name: "Youssef Nader",
      title: "Machine Learning Researcher",
      href: "https://youssefnader.com/",
    },
    {
      name: "Hendrik Schilling",
      title: "Computer Vision & AI Expert, PhD",
      href: "https://www.linkedin.com/in/dr-hendrik-schilling-a2019418a",
    },
    {
      name: "Elian Rafael Dal Prá",
      title: "ML Intern & Annotation Specialist",
      href: "https://twitter.com/elianrafaeldp",
    },
    {
      name: "Johannes Rudolph",
      title: "Platform Engineer",
      href: "https://blog.virtual-void.net/",
    },
    {
      name: "Forrest McDonald",
      title: "Software Engineer",
      href: "https://www.linkedin.com/in/forrest-mcdonald-a80b9885/",
    },
    {
      name: "David Josey",
      title: "ML Annotation Team Lead, PhD",
      href: "https://www.linkedin.com/in/davidsjosey/",
    },
    {
      name: "Eric Thvedt",
      title: "Annotation Specialist",
      href: "https://www.linkedin.com/in/eric-thvedt/",
    },
    {
      name: "Kendra Brown",
      title: "Annotation Specialist",
      href: "https://darthkendraresearch.wordpress.com/",
    },
    {
      name: "Konrad Rosenberg",
      title: "Annotation Specialist",
      href: "https://twitter.com/germanicgems",
    },
    {
      name: "Sarah Morejohn",
      title: "Annotation Specialist",
      href: "https://www.linkedin.com/in/sarah-morejohn-1140b049/",
    },
    {
      name: "Techjays",
      title: "Annotation Services",
      href: "https://www.techjays.com/",
    },
  ],
  educe: [
    {
      name: "Brent Seales",
      title: "Principal Investigator, Professor of Computer Science",
      href: "https://educelab.engr.uky.edu/w-brent-seales",
    },
    {
      name: "Seth Parker",
      title: "Research Manager",
      href: "https://www2.cs.uky.edu/dri/seth-parker/",
    },
    {
      name: "Christy Chapman",
      title: "Research & Partnership Manager",
      href: "https://educelab.engr.uky.edu/christy-chapman",
    },
    {
      name: "Mami Hayashida",
      title: "Research Staff",
      href: "https://www.ccs.uky.edu/about-ccs/staff-directory/mami-hayashida/",
    },
    {
      name: "James Brusuelas",
      title: "Associate Professor of Classics",
      href: "https://mcl.as.uky.edu/users/jbr454",
    },
    {
      name: "Beth Lutin",
      title: "College Business Analyst",
      href: "https://www.engr.uky.edu/directory/lutin-elizabeth",
    },
    {
      name: "Roger Macfarlane",
      title: "Professor of Classical Studies",
      href: "https://hum.byu.edu/directory/roger-macfarlane",
    },
  ],
  alumni: [
    {
      name: "JP Posma",
      title: "Project Lead",
      href: "https://janpaulposma.nl/",
    },
    {
      name: "Ben Kyles",
      title: "Segmentation Team Lead",
      href: "https://twitter.com/ben_kyles",
    },
    {
      name: "Julian Schilliger",
      title: "Software Engineer",
      href: "https://www.linkedin.com/in/julian-schilliger-963b21294/",
    },
    {
      name: "Adrionna Fey",
      title: "Annotation Specialist",
      href: "https://twitter.com/Meadowsnax1",
    },
    {
      name: "Cooper Miller",
      title: "Annotation Specialist",
      href: "https://kcm.sh/",
    },
    {
      name: "Raymond Gasper",
      title: "Annotation Specialist",
      href: "https://www.linkedin.com/in/raymond-james-gasper/",
    },
    {
      name: "Sergei Pnev",
      title: "Annotation Specialist",
      href: "www.linkedin.com/in/sergey-pnev",
    },
    {
      name: "Daniel Havíř",
      title: "Machine Learning",
      href: "https://danielhavir.com/",
    },
    {
      name: "Ian Janicki",
      title: "Design",
      href: "https://ianjanicki.com/",
    },
    {
      name: "Chris Frangione",
      title: "Prizes",
      href: "https://www.linkedin.com/in/chrisfrangione/",
    },
    {
      name: "Garrett Ryan",
      title: "Classics",
      href: "https://toldinstone.com/",
    },
    {
      name: "Dejan Gotić",
      title: "3D Animator",
      href: "https://www.instagram.com/dejangotic_constructology/",
    },
    {
      name: "Jonny Hyman",
      title: "2D Animator",
      href: "https://jonnyhyman.com/",
    },
  ],
  papyrology: [
    {
      name: "Federica Nicolardi (Lead Papyrologist)",
      title:
        "Assistant Professor of Papyrology, University of Naples Federico II",
      href: "https://www.docenti.unina.it/federica.nicolardi",
    },
    {
      name: "Marzia D'Angelo",
      title:
        "Postdoctoral Fellow in Papyrology, University of Naples Federico II",
      href: "https://unina.academia.edu/MDAngelo",
    },
    {
      name: "Kilian Fleischer",
      title:
        "Research Director and Papyrologist, University of Tübingen and CNR",
      href: "https://www.klassphil.uni-wuerzburg.de/team/pd-dr-kilian-fleischer/",
    },
    {
      name: "Alessia Lavorante",
      title:
        "Postdoctoral Fellow in Papyrology, University of Naples Federico II",
      href: "https://unina.academia.edu/AlessiaLavorante",
    },
    {
      name: "Michael McOsker",
      title: "Researcher, University College London",
      href: "https://profiles.ucl.ac.uk/97051-michael-mcosker",
    },
    {
      name: "Maria Chiara Robustelli",
      title: "Postdoctoral Fellow in Papyrology, University of Naples Federico II",
      href: "https://unina.academia.edu/mrobustelli",
    },
    {
      name: "Claudio Vergara",
      title:
        "Postdoctoral Fellow in Papyrology, University of Naples Federico II",
      href: "https://unina.academia.edu/ClaudioVergara",
    },
    {
      name: "Rossella Villa",
      title: "Research Assistant in Papyrology, University of Salerno",
      href: "https://salerno.academia.edu/RossellaVilla",
    },
  ],
  papyrologyAdvisors: [
    {
      name: "Daniel Delattre",
      title: "Emeritus Research Director and Papyrologist, CNRS and IRHT",
      href: "https://www.irht.cnrs.fr/fr/annuaire/delattre-daniel",
    },
    {
      name: "Gianluca Del Mastro",
      title:
        "Professor of Papyrology, l'Università della Campania «L. Vanvitelli»",
      href: "https://www.facebook.com/GianlucaDelMastroSindaco",
    },
    {
      name: "Robert Fowler",
      title:
        "Fellow of the British Academy;  Professor Emeritus of Classics, Bristol University",
      href: "https://www.thebritishacademy.ac.uk/fellows/robert-fowler-FBA/",
    },
    {
      name: "Richard Janko",
      title:
        "Fellow of the American Academy of Arts and Sciences; Professor of Classics, University of Michigan",
      href: "https://lsa.umich.edu/classics/people/departmental-faculty/rjanko.html",
    },
    {
      name: "Federica Nicolardi",
      title:
        "Assistant Professor of Papyrology, University of Naples Federico II",
      href: "https://www.docenti.unina.it/federica.nicolardi",
    },
    {
      name: "Tobias Reinhardt",
      title:
        "Corpus Christi Professor of the Latin Language and Literature, Oxford",
      href: "https://www.classics.ox.ac.uk/people/professor-tobias-reinhardt",
    },
  ],
};

const partners = [
  {
    icon: "/img/landing/educe.svg",
    href: "https://educelab.engr.uky.edu/",
  },
  {
    icon: "/img/landing/institute.svg",
    href: "https://www.institutdefrance.fr/en/home/",
  },
  {
    icon: "/img/landing/diamond.svg",
    href: "https://www.diamond.ac.uk/",
  },
  {
    icon: "/img/landing/biblioteca.svg",
    href: "https://www.bnnonline.it/",
  },
  {
    icon: "/img/landing/getty.svg",
    href: "https://www.getty.edu/",
  },
  {
    icon: "/img/landing/kaggle.svg",
    href: "https://www.kaggle.com/",
  },
  {
    icon: "/img/landing/panua.svg",
    href: "https://panua.ch/",
  },
];

const educelabFunders = [
  {
    name: "The National Science Foundation",
    href: "https://www.nsf.gov/",
  },
  {
    name: "The National Endowment for the Humanities",
    href: "https://www.neh.gov/",
  },
  {
    name: "The Andrew W. Mellon Foundation",
    href: "https://www.mellon.org/",
  },
  {
    name: "The Digital Restoration Initiative",
    href: "https://www2.cs.uky.edu/dri/",
  },
  {
    name: "The Arts & Humanities Research Council",
    href: "https://www.ukri.org/councils/ahrc/",
  },
  {
    name: "The Lighthouse Beacon Foundation — Stanley and Karen Pigman",
    href: undefined,
  },
  {
    name: "John & Karen Maxwell",
    href: undefined,
  },
  {
    name: "Lee & Stacie Marksbury",
    href: undefined,
  },
];

const Story = ({ story, index }) => (
  <section
    id={`story-section-${index}`}
    className="mb-30 md:h-full h-auto"
    style={{
      opacity: 1,
      ...(index === 0 && {
        background:
          "linear-gradient(360deg, rgba(28, 26, 29, 0) 59.11%, #1C1A1D 99.36%)",
      }),
    }}
  >
    <div className="container mx-auto z-30 relative">
      <div className="py-10 max-w-4xl">
        <h1
          className="text-3xl md:text-6xl font-black mb-2 leading-none tracking-tighter"
          id={story.anchor}
        >
          <span
            style={{
              background:
                "radial-gradient(53.44% 245.78% at 13.64% 46.56%, #F5653F 0%, #D53A17 100%)",
              WebkitBackgroundClip: "text",
              WebkitTextFillColor: "transparent",
              backgroundClip: "text",
              textFillColor: "transparent",
            }}
          >
            {story.date}
          </span>
          <br />
          {story.text}
        </h1>
        <div className="md:text-xl text-lg font-medium mb-8 text-[rgba(255,255,255,0.7)] leading-6 tracking-tight whitespace-pre-line">
          {story.description}
        </div>
      </div>
    </div>
  </section>
);

const StoryBackground = ({ story, index }) => (
  <div
    className="fixed inset-0 z-0"
    id={`story-image-${index}`}
    style={{
      background: `url(${story.background})`,
      backgroundSize: "60%",
      backgroundPosition: "center right",
      backgroundRepeat: "no-repeat",
      opacity: 0,
    }}
  />
);

const Winners = ({ winners, large }) => (
  <div className={`flex ml-3 ${large ? "h-10" : "h-8"}`}>
    {winners.map((winner, i) => (
      <React.Fragment key={i}>
        <div className="-ml-3" style={{ zIndex: 100 - i }}>
          <img
            src={winner.image}
            className={`${large ? "h-10" : "h-8"} rounded-full border-2 ${
              large ? "border-[#272222]" : "border-[#1C1A1D]"
            } border-solid`}
          />
        </div>
      </React.Fragment>
    ))}
  </div>
);

const Prize = ({ prize }) => (
  <a
    href={!prize.tba ? prize.href : "#"}
    className={`text-white hover:text-white hover:no-underline group ${
      prize.tba ? "opacity-40" : ""
    }`}
  >
    <div
      className={`flex flex-col bg-[#131114bf] border border-solid h-full ${
        prize.bannerImage ? "" : "md:p-6 p-4"
      } rounded-2xl relative ${
        prize.winners ? `border-[#F5653F40]` : `border-[#FFFFFF40]`
      }  hover:-translate-y-2 transition-transform ease-in-out duration-300 overflow-hidden`}
      style={{
        boxShadow:
          "0px 2.767px 2.214px 0px rgba(0, 0, 0, 0.09), 0px 6.65px 5.32px 0px rgba(0, 0, 0, 0.13), 0px 12.522px 10.017px 0px rgba(0, 0, 0, 0.16), 0px 22.336px 17.869px 0px rgba(0, 0, 0, 0.19), 0px 41.778px 33.422px 0px rgba(0, 0, 0, 0.23), 0px 100px 80px 0px rgba(0, 0, 0, 0.32)",
      }}
    >
      <div className={`${!prize.bannerImage ? "" : "md:p-6 p-4"}`}>
        {prize.winners && (
          <p
            className={`font-bold uppercase text-[var(--ifm-color-primary)] ${
              prize.bannerImage ? "!mb-2 text-sm" : "!mb-0 text-xs"
            }`}
          >
            won
          </p>
        )}
        <h2
          className={`${
            prize.bannerImage ? "text-2xl lg:text-4xl " : "text-xl md:text-2xl "
          } font-black !mb-0 !leading-none tracking-tighter !my-0`}
        >
          {prize.title} {prize.tba && <span className="opacity-60">TBA</span>}
        </h2>
        <h3
          className={`${
            prize.bannerImage ? "text-xl lg:text-3xl " : "text-lg md:text-2xl"
          } font-black !leading-none tracking-tighter !mb-0`}
          style={{
            background:
              "radial-gradient(53.44% 245.78% at 13.64% 46.56%, #F5653F 0%, #D53A17 100%)",
            WebkitBackgroundClip: "text",
            WebkitTextFillColor: "transparent",
            backgroundClip: "text",
            textFillColor: "transparent",
          }}
        >
          {prize.prizeMoney}
        </h3>
        {(!prize.winners || prize.bannerImage) && !prize.tba && (
          <div className="pt-2">
            <AnimatedArrow
              text={prizes.winners ? "Read the announcement" : "Learn more"}
            />
          </div>
        )}
      </div>
      {prize.winners && !prize.bannerImage && (
        <div className="flex gap-2 items-center pt-2">
          <Winners
            winners={prize.winners}
            winnersLabel={prize.winnersLabel}
            large={prize.bannerImage ? true : false}
          />
          <h3 className="!mb-0 opacity-60">
            {prize.winnersLabel
              ? prize.winnersLabel
              : `${prize.winners.length} Winners`}
          </h3>
        </div>
      )}
      {!prize.winners && (
        <p className="flex-1 md:text-xl text-lg font-medium opacity-60 leading-none tracking-tight">
          {prize.description}
        </p>
      )}
      {prize.bannerImage && (
        <div className="bg-[#232222] h-full flex flex-col justify-between">
          <div className="flex gap-2 items-center md:px-6 px-4 py-3">
            <Winners
              winners={prize.winners}
              large={prize.bannerImage ? true : false}
            />
            <h3 className="!mb-0 opacity-60">
              {prize.winnersLabel
                ? prize.winnersLabel
                : `${prize.winners.length} Winners`}
            </h3>
          </div>
          <img
            src={prize.bannerImage}
            className={`block max-h-16 ${
              prize.href === "/firstletters"
                ? "object-contain object-right -mt-6"
                : "object-cover"
            }`}
          />
        </div>
      )}
    </div>
  </a>
);

const Creator = ({ creator }) => (
  <a
    href={creator.href}
    className="text-white hover:text-white hover:no-underline"
  >
    <div className="flex items-center gap-3 rounded-2xl bg-[#131114bf] border border-solid h-full md:p-6 p-4 border-[#FFFFFF40] hover:bg-[#292525d6] transition-color ease-in-out duration-300">
      <img
        src={creator.image}
        className="md:w-20 md:h-20 w-12 h-12 rounded-full saturate-0"
      />
      <h2 className="text-2xl md:text-4xl font-black !mb-0 !leading-[90%] tracking-tighter !my-0">
        {creator.name}
      </h2>
    </div>
  </a>
);

const Sponsor = ({ sponsor }) => {
  let image = "";
  let name = "";
  let amount = "";
  let padding = "";
  let radius = "";
  let level = "";
  let multipleScale = "";
  if (sponsor.amount >= 200000) {
    image = "w-16 h-16";
    name = "text-xl md:text-2xl";
    amount = "text-lg md:text-xl";
    padding = "md:p-3 p-2";
    radius = "rounded-2xl";
    level = "text-[#E8A42F]";
    multipleScale = "xl:scale-[0.65]";
  } else if (sponsor.amount >= 50000 && sponsor.amount < 200000) {
    image = "w-12 h-12";
    name = "text-lg md:text-xl";
    amount = "text-sm md:text-md";
    padding = "p-2";
    radius = "rounded-xl";
    level = "text-[#8658ED]";
    multipleScale = "xl:scale-[0.65]";
  } else {
    image = "w-8 h-8";
    name = "text-sm md:text-md";
    amount = "text-xs";
    padding = "md:p-2 p-1";
    radius = "rounded-lg";
    level = "text-[#F5653F]";
    multipleScale = "xl:scale-[0.85]";
  }

  return (
    <a
      href={sponsor.href}
      className={`text-white hover:text-white hover:no-underline`}
      target="_blank"
    >
      <div
        className={`${padding} ${radius} flex items-center gap-2 bg-[#131114bf] border border-solid h-full border-[#FFFFFF40] hover:bg-[#292525d6] transition-color ease-in-out duration-300`}
      >
        {!sponsor.image ? (
          <></>
        ) : typeof sponsor.image === "object" ? (
          <div className={`flex ${multipleScale} origin-left`}>
            {sponsor.image.map((img, i) => (
              <img
                key={i}
                src={img}
                className={`${image} ${
                  i === 1 ? "-ml-3" : ""
                } rounded-full saturate-0 border-2 border-solid border-[#272222]`}
                style={{ zIndex: 100 - i }}
              />
            ))}
          </div>
        ) : (
          <img
            src={sponsor.image}
            className={`${image} rounded-full saturate-0`}
          />
        )}
        <div className="flex flex-col z-[1000]">
          <h2
            className={`${name} font-black !mb-0 !leading-[90%] tracking-tighter !my-0`}
          >
            {sponsor.name}
          </h2>
          <h3
            className={`${amount} ${level} font-bold !leading-none tracking-tighter !mb-0`}
          >
            {new Intl.NumberFormat("en-US", {
              style: "currency",
              currency: "USD",
              maximumSignificantDigits: 10,
            }).format(sponsor.amount)}
          </h3>
        </div>
      </div>
    </a>
  );
};

const Link = ({ link }) => (
  <div>
    <a
      className=" text-white hover:no-underline inline-block fit-content"
      href={link.href}
    >
      <h3 className="mb-0 text-xl font-medium">
        {link.name}&nbsp;&nbsp;
        <span className="opacity-70">{link.title}</span>
      </h3>
    </a>
  </div>
);

const crossfadeHeight = 150;

const getBounds = (div) => div.getBoundingClientRect();
const getOpacity = ({ y, height }) =>
  Math.max(0, Math.min(1, (1 - Math.abs(y) / height) * 1.5));
const getBackgroundOpacity = ({ y, height }) => {
  if (y > -height - crossfadeHeight && y < crossfadeHeight) {
    // top transition
    if (y < crossfadeHeight && y > -crossfadeHeight) {
      return (1 - y / crossfadeHeight) / 2;
      // bottom transition
    } else if (y < -height + crossfadeHeight && y > -height - crossfadeHeight) {
      return (y + height + crossfadeHeight) / (crossfadeHeight * 2);
    } else {
      return 1;
    }
  } else {
    return 0;
  }
};

const autoPlay = (ref) =>
  ref &&
  ref.current
    .play()
    .then(() => {})
    .catch((err) => {
      // Video couldn't play, low power play button showing.
    });

const AnimatedArrow = ({ text, button }) => (
  <div className={`flex ${button ? "" : "opacity-60"} text-sm`}>
    <div className="hidden sm:block uppercase font-bold tracking-wider mr-1 group-hover:mr-3 transition-all ease-in-out duration-300">
      {text}
    </div>
    <div className="block sm:hidden uppercase font-bold tracking-wider mr-1">
      {text}
    </div>
    <img
      src={
        button
          ? "/img/landing/arrow-right.svg"
          : "/img/landing/arrow-right-white.svg"
      }
    />
  </div>
);

// export default BeforeAfter;

const LargeAnimatedArrow = ({ text, button }) => (
  <div className={`flex ${button ? "" : "opacity-60"} text-l`}>
    <div className="hidden sm:block uppercase font-bold tracking-wider mr-1 group-hover:mr-3 transition-all ease-in-out duration-300">
      {text}
    </div>
    <div className="block sm:hidden uppercase font-bold tracking-wider mr-1">
      {text}
    </div>
    <img
      src={
        button
          ? "/img/landing/arrow-right.svg"
          : "/img/landing/arrow-right-white.svg"
      }
    />
  </div>
);

const ChallengeBox = ({
  title,
  children,
  linkText,
  href,
  imageSrc,
  imagePosition = "right",
}) => {
  // When imagePosition is "top"
  if (imagePosition === "top") {
    return (
      <div className="w-full flex flex-col bg-[#131114bf] p-5 rounded-xl justify-between border border-[#FFFFFF20]">
        {/* Responsive image container:
            On mobile, height is natural; on md+ screens, fixed height */}
        <div className="mb-4 md:h-48">
          {imageSrc &&
            // If imageSrc is a string, render an <img>; otherwise assume it's a component
            (typeof imageSrc === "string" ? (
              <img
                src={imageSrc}
                alt="Scroll representation"
                className="rounded-lg w-full h-full object-cover"
              />
            ) : (
              <div className="w-full">{imageSrc}</div>
            ))}
        </div>
        {/* Title */}
        <div className="h-12 flex items-center">
          <h2 className="text-white text-2xl font-bold">{title}</h2>
        </div>
        {/* Divider */}
        <div className="h-px bg-[#FFFFFF20] mb-4" />
        {/* Content */}
        <div className="flex-grow">{children}</div>
        {/* Link */}
        <a href={href} className="mt-4 block group">
          <div className="group-hover:-translate-y-2 transition-transform ease-in-out duration-300">
            <LargeAnimatedArrow text={linkText} />
          </div>
        </a>
      </div>
    );
  }

  // When imagePosition is "bottom"
  if (imagePosition === "bottom") {
    return (
      <div className="w-full flex flex-col bg-[#131114bf] p-5 mb-5 rounded-xl justify-between border border-[#FFFFFF20]">
        <b className="text-white text-2xl mb-3">{title}</b>
        <div className="h-px bg-[#FFFFFF20] mb-4" />
        {children}
        <a href={href} className="mt-auto group">
          <div className="group-hover:-translate-y-2 transition-transform ease-in-out duration-300">
            <LargeAnimatedArrow text={linkText} />
          </div>
        </a>
        {imageSrc && (
          <div className="mt-4 w-full">
            {typeof imageSrc === "string" ? (
              <img
                src={imageSrc}
                alt="Scroll representation"
                className="rounded-lg w-full h-auto"
              />
            ) : (
              <div className="w-full">{imageSrc}</div>
            )}
          </div>
        )}
      </div>
    );
  }

  // When imagePosition is "left"
  if (imagePosition === "left") {
    return (
      <div className="w-full flex flex-col bg-[#131114bf] p-5 mb-5 rounded-xl justify-between border border-[#FFFFFF20]">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="w-full">
            {imageSrc &&
              (typeof imageSrc === "string" ? (
                <img
                  src={imageSrc}
                  alt="Scroll representation"
                  className="rounded-lg w-full h-auto"
                />
              ) : (
                <div className="w-full">{imageSrc}</div>
              ))}
          </div>
          <div className="flex flex-col">
            <b className="text-white text-2xl mb-3">{title}</b>
            <div className="h-px bg-[#FFFFFF20] mb-4" />
            {children}
            <a href={href} className="mt-auto group">
              <div className="group-hover:-translate-y-2 transition-transform ease-in-out duration-300">
                <LargeAnimatedArrow text={linkText} />
              </div>
            </a>
          </div>
        </div>
      </div>
    );
  }

  // Default layout: imagePosition "right"
  // On mobile: single-column grid with the image on top;
  // On md+ screens: two columns with text on the left and image on the right.
  return (
    <div className="w-full flex flex-col bg-[#131114bf] p-5 mb-5 rounded-xl justify-between border border-[#FFFFFF20]">
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* Text and Link Container */}
        <div className="order-2 md:order-1 flex flex-col">
          <b className="text-white text-2xl mb-3">{title}</b>
          <div className="h-px bg-[#FFFFFF20] mb-4" />
          {children}
          <a href={href} className="mt-auto group">
            <div className="group-hover:-translate-y-2 transition-transform ease-in-out duration-300">
              <LargeAnimatedArrow text={linkText} />
            </div>
          </a>
        </div>
        {/* Image Container */}
        <div className="order-1 md:order-2">
          {imageSrc && (
            <div className="w-full">
              {typeof imageSrc === "string" ? (
                <img
                  src={imageSrc}
                  alt="Scroll representation"
                  className="rounded-lg w-full h-auto"
                />
              ) : (
                <div className="w-full">{imageSrc}</div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};


const BuildingBlock = ({ title, description, showDividerMobile = true, showDividerDesktop = true }) => (
  <div className="relative px-3">
    {(showDividerMobile || showDividerDesktop) && (
      <div className={`absolute right-0 top-0 bottom-0 w-px bg-orange-600 ${showDividerMobile ? 'block' : 'hidden'} ${showDividerDesktop ? 'md:block' : 'md:hidden'}`} />
    )}
    <b className="block mb-2">{title}</b>
    <p className="text-sm">{description}</p>
  </div>
);

const App = () => {
  const tabData = [{ label: "Tab 1" }, { label: "Tab 2" }, { label: "Tab 3" }];

  return (
    <div className="App">
      <h1 className="geeks">GeeksforGeeks</h1>
      <h1>React Tabs Example</h1>
      <Tabs tabs={tabData} />
    </div>
  );
};

export function Landing() {
  useBrokenLinks().collectAnchor("sponsors");
  useBrokenLinks().collectAnchor("educelab-funders");
  useBrokenLinks().collectAnchor("our-story");

  const heroVideo = useRef(null);
  const unrollVideo = useRef(null);
  // const mlVideo = useRef(null);
  // const xrayVideo = useRef(null);

  useEffect(() => {
    if (!globalThis.window) {
      return;
    }
    const storyDivs = Array.from(
      document.querySelectorAll("[id^='story-section']"),
    );
    const imageDivs = Array.from(
      document.querySelectorAll("[id^='story-image']"),
    );
    const onScroll = () => {
      const storyBounds = storyDivs.map((div) => getBounds(div));
      const backgroundOpacities = storyBounds.map((bounds) =>
        getBackgroundOpacity({
          y: bounds.y - window.innerHeight / 2,
          height: bounds.height,
        }),
      );
      imageDivs.forEach(
        (story, index) =>
          (story.style.opacity = backgroundOpacities[index] * 0.4),
      );
    };
    window.addEventListener("scroll", onScroll);
    return () => window.removeEventListener("scroll", onScroll);
  }, []);

  useEffect(() => {
    autoPlay(heroVideo);
    autoPlay(unrollVideo);
    // autoPlay(mlVideo);
    // autoPlay(xrayVideo);
  }, []);

  return (
    <>
      <div className="absolute inset-0 z-0 md:block hidden">
        {stories({ unrollVideo }).map((s, index) => (
          <StoryBackground story={s} key={s.date} index={index} />
        ))}
      </div>
      <div className="text-white ">
        <div className="z-20 relative">
          {/* Hero */}
          <section>
            <div className="container mx-auto z-20 relative mb-12">
              <div className="md:pt-20 pt-8 mb-4">
                <h1 className="text-4xl md:text-7xl font-black !mb-4 tracking-tighter mix-blend-exclusion !leading-[90%] transition-opacity">
                  <div className="max-w-3xl">
                    Resurrect an ancient library from the ashes of a volcano.
                  </div>
                  <span
                    className="text-3xl md:text-6xl drop-shadow-lg"
                    style={{
                      background:
                        "radial-gradient(53.44% 245.78% at 13.64% 46.56%, #F5653F 0%, #D53A17 100%)",
                      WebkitBackgroundClip: "text",
                      WebkitTextFillColor: "transparent",
                      backgroundClip: "text",
                      textFillColor: "transparent",
                    }}
                  >
                    <span className="whitespace-nowrap">Win Prizes.&nbsp;</span>
                    &nbsp;
                    <span className="whitespace-nowrap">
                      Make History.&nbsp;
                    </span>
                  </span>
                </h1>
                <p className="max-w-lg md:text-xl text-lg font-medium mb-8 !leading-[110%] tracking-tight">
                  <span className="opacity-80 md:opacity-60">
                    Vesuvius Challenge is a machine learning, computer vision,
                    and geometry competition that is{" "}
                  </span>
                  <span className="opacity-100">
                    <a href="grandprize">reading</a>&nbsp;
                  </span>
                  <span className="opacity-80 md:opacity-60">
                    the carbonized Herculaneum scrolls & has awarded $1,500,000 in prizes.
                  </span>
                  <br />
                  <br />
                  <span className="opacity-80 md:opacity-60">
                    Our current challenge is to grow from a few passages to
                    entire scrolls.&nbsp;
                  </span>
                  <span className="opacity-100">
                    <a href="/get_started">Join the community</a>&nbsp;
                  </span>
                  <span className="opacity-80 md:opacity-60">
                    to win prizes and make history.
                  </span>
                  <span className="opacity-80 md:opacity-60">&nbsp;Also:</span>
                  <span className="opacity-100">
                    <a href="jobs">&nbsp;we're hiring!</a>
                  </span>
                </p>
              </div>

              <div className="grid items-start max-w-8xl">
                <LatestPosts />

                <div className="flex-wrap z-10 pt-1">
                  <div className="grid grid-cols-1">
                    <ChallengeBox
                      title="Open Problem: Representation"
                      linkText="Scan the Surface"
                      href="/unwrapping"
                      imageSrc={
                        <BeforeAfter
                          beforeImage="/img/data/rep_raw_10037.png"
                          afterImage="/img/data/rep_norms_10037.png"
                        />
                      }
                      imagePosition="right"
                    >
                      <p className="">
                        Carbonized and crushed under pyroclastic flow and
                        debris, the scrolls are in rough shape. Tracing the 3D
                        sheets through these damaged scrolls is nearly
                        impossible in the raw scan data. More structured
                        representations, like those obtained with semantic
                        segmentation, simplify downstream tasks significantly.
                      </p>
                      <p className=" pt-16">
                        Related skills: image annotation, computer vision,
                        machine learning, medical imaging
                      </p>
                    </ChallengeBox>

                    <ChallengeBox
                      title="Open Problem: Geometric Reconstruction"
                      linkText="Chart the Path"
                      href="/segmentation"
                      imageSrc={
                        <BeforeAfter
                          beforeImage="/img/data/raw_pred.png"
                          afterImage="/img/data/patches.png"
                        />
                      }
                      imagePosition="right"
                    >
                      <p className="pb-5">
                        A better image representation alone does not an unrolled
                        scroll make. We need methods to better map the surfaces,
                        stitch them where necessary, and extract them into
                        readable sheets of papyrus.

                        For a primer on current autosegmentation methods and their progress, read the <a href="/unwrapping"> Virtual Unwrapping document.</a>
                      </p>
                      <p className=" pt-16">
                        Related skills: geometry processing, computer vision,
                        machine learning, optimization
                      </p>
                    </ChallengeBox>

                    <ChallengeBox
                      title="Open Problem: Ink Detection"
                      linkText="Find a Letter"
                      href="/tutorial5"
                      imageSrc={
                        <BeforeAfter
                          beforeImage="/img/ink/51002_crop/32.jpg"
                          afterImage="/img/ink/51002_crop/prediction.jpg"
                        />
                      }
                      imagePosition="right"
                    >
                      <p className="">
                        We've so far recovered text from just two of our five
                        scrolls. Is the ink fundamentally different in others?
                        Is the papyrus surface? We're not yet sure. We are
                        certain though that if it ever existed, it can be
                        detected.
                      </p>
                      <p className="pt-16">
                        Related skills: image annotation, computer vision,
                        machine learning, pattern recognition
                      </p>
                    </ChallengeBox>
                  </div>
                </div>
              </div>
              <div
                className="mt-2 pt-2 pb-0 relative rounded-2xl border-solid text-white border border-[#FFFFFF20] bg-[#131114bf]"
                style={{
                  boxShadow:
                    "0px 2.767px 2.214px 0px rgba(0, 0, 0, 0.09), 0px 6.65px 5.32px 0px rgba(0, 0, 0, 0.13), 0px 12.522px 10.017px 0px rgba(0, 0, 0, 0.16), 0px 22.336px 17.869px 0px rgba(0, 0, 0, 0.19), 0px 41.778px 33.422px 0px rgba(0, 0, 0, 0.23), 0px 100px 80px 0px rgba(0, 0, 0, 0.32)",
                }}
              >
                <h3 className="text-2xl text-white pt-2 pb-3 text-center">
                  What We're Building Towards
                </h3>

                <div className="grid grid-cols-2 md:grid-cols-4 gap-4 md:gap-5 px-6 pb-3">
                  <BuildingBlock
                    title="Accurate Surface Representation"
                    description="We lack the accuracy to make the meshing step as simple as it could be."
                    showDividerMobile={true}
                    showDividerDesktop={true}
                  />
                  <BuildingBlock
                    title="Generalizable Ink Detection"
                    description="Ink has been found in two scrolls, but remains elusive in our other scrolls."
                    showDividerMobile={false}
                    showDividerDesktop={true}
                  />
                  <BuildingBlock
                    title="High Quality Annotations"
                    description="We need an abundance of high-quality annotations."
                    showDividerMobile={true}
                    showDividerDesktop={true}
                  />
                  <BuildingBlock
                    title="Robust Meshing"
                    description="Methods that function where Surface Representation is unreliable are needed."
                    showDividerMobile={false}
                    showDividerDesktop={false}
                  />
                </div>
              </div>

              <div className="pt-8 mb-4">
                <p className="max-w-lg md:text-xl text-lg font-medium mb-8 !leading-[110%] tracking-tight">
                  <span id="our-story" className=" opacity-80 md:opacity-60">
                    Our story ↓
                  </span>
                </p>
              </div>
            </div>
            <div
              className="absolute inset-0 h-[75vh] z-10"
              style={{
                background:
                  "linear-gradient(90deg, rgba(28, 26, 29, 0.8) 20%, rgba(28, 26, 29, 0) 80%),linear-gradient(0deg, #1C1A1D 1%, rgba(28, 26, 29, 0) 30%)",
              }}
            />
            <div className="absolute inset-0 h-[75vh] z-0">
              <video
                // autoPlay
                playsInline
                loop
                muted
                poster="/img/landing/vesuvius.webp"
                className="w-full h-full object-cover object-[45%]"
                ref={heroVideo}
              >
                <source
                  src="img/landing/vesuvius-flipped-min.webm"
                  type="video/webm"
                />
              </video>
            </div>
          </section>
          {/* Stories */}
          {stories({ unrollVideo }).map((s, index) => (
            <Story story={s} key={s.date} index={index} />
          ))}
          {/* Prize */}
          <section className="mb-24 md:mb-36">
            <div className="container mx-auto z-30 relative">
              <div className="flex flex-col py-8 md:py-16 ">
                <h1 className="text-3xl md:text-6xl font-black !mb-2 leading-none tracking-tighter">
                  <span
                    className="font-black leading-none tracking-tighter mb-0"
                    style={{
                      background:
                        "radial-gradient(53.44% 245.78% at 13.64% 46.56%, #F5653F 0%, #D53A17 100%)",
                      WebkitBackgroundClip: "text",
                      WebkitTextFillColor: "transparent",
                      backgroundClip: "text",
                      textFillColor: "transparent",
                    }}
                  >
                    {new Date().getFullYear()} AD
                  </span>
                  <br />
                  The Challenge Continues
                </h1>
                <p className="max-w-xl md:text-xl text-lg font-medium !mb-8 md:w-full w-4/5  !leading-[110%] tracking-tight opacity-60">
                  Vesuvius Challenge moves onto its next stage of reading multiple entire scrolls.
                  Read more about the prizes below, and on how
                  they contribute towards the{" "}
                  <a href="master_plan">The Master Plan</a>.
                </p>
                <div className="flex flex-col gap-3 max-w-7xl">
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-3 max-w-6xl">
                    {prizes
                      .filter((p) => !p.winners)
                      .map((p, i) => (
                        <Prize prize={p} key={i} />
                      ))}
                  </div>
                </div>
              </div>
              <div className="pt-10 md:pt-20 max-w-3xl">
                <h1 className="text-4xl md:text-7xl font-black !mb-2 leading-none tracking-tighter">
                  Awarded Prizes
                </h1>
                <p className="max-w-xl md:text-xl text-lg font-medium !mb-8 md:w-full w-4/5  !leading-[110%] tracking-tight opacity-60">
                  Incredible teams of engineers are helping us unlock these
                  secrets, providing unprecedented access to scrolls that have
                  not been read in two millennia. Learn more about their
                  accomplishments.
                </p>
              </div>
              <div className="flex flex-col gap-3 max-w-7xl">
                <div className="grid grid-cols-1 md:grid-cols-3 gap-3 max-w-6xl">
                  {prizes
                    .filter((p) => p.winners && p.bannerImage)
                    .map((p, i) => (
                      <Prize prize={p} key={i} />
                    ))}
                </div>
                <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-3 max-w-6xl">
                  {prizes
                    .filter((p) => p.winners && !p.bannerImage)
                    .map((p, i) => (
                      <Prize prize={p} key={i} />
                    ))}
                </div>
              </div>
            </div>
          </section>
          {/* Team */}
          <section>
            <div className="container mx-auto z-30 relative">
              <div className="mb-6 md:mb-24 max-w-6xl">
                <h1 className="mb-16 text-4xl md:text-7xl font-black leading-none tracking-tighter ">
                  Created By
                </h1>
                <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 auto-rows-fr gap-2">
                  {creators.map((c, i) => (
                    <Creator creator={c} key={i} />
                  ))}
                </div>
              </div>
              <div className="mb-6 md:mb-10 max-w-6xl">
                <h1
                  className="mb-16 text-4xl md:text-7xl font-black leading-none tracking-tighter "
                  name="sponsors"
                  id="sponsors"
                >
                  Sponsors
                </h1>
                <h2 className="text-3xl md:text-5xl text-[#E8A42F]">Caesars</h2>
                <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-4 auto-rows-fr  gap-2 pb-8">
                  {sponsors
                    .filter((s) => s.amount >= 200000)
                    .map((s, i) => (
                      <Sponsor sponsor={s} key={i} />
                    ))}
                </div>
                <h2 className="text-3xl md:text-5xl text-[#8658ED]">
                  Senators
                </h2>
                <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 2xl:grid-cols-5 auto-rows-fr gap-2 pb-8">
                  {sponsors
                    .filter((s) => s.amount >= 50000 && s.amount < 200000)
                    .map((s, i) => (
                      <Sponsor sponsor={s} key={i} />
                    ))}
                </div>
                <h2 className="text-3xl md:text-5xl text-[#F5653F]">
                  Citizens
                </h2>
                <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 2xl:grid-cols-6 gap-2">
                  {sponsors
                    .filter((s) => s.amount < 50000)
                    .map((s, i) => (
                      <Sponsor sponsor={s} key={i} />
                    ))}
                </div>
                <div className="flex justify-center py-8">
                  <a href="https://donate.stripe.com/aEUg101vt9eN8gM144">
                    <button
                      className="px-4 py-3 uppercase font-bold rounded-full border-[#F5653F] border-solid text-[#F5653F] bg-transparent cursor-pointer group
                    "
                    >
                      <AnimatedArrow text="Become a sponsor" button />
                    </button>
                  </a>
                </div>
              </div>
              <div className="py-10">
                <h1 className="hidden md:block text-4xl md:text-7xl font-black leading-none tracking-tighter ">
                  Team
                </h1>
                <div className="flex flex-wrap">
                  <div className="flex-1 flex-col lg:gap-0 gap-2 mt-8 min-w-[100%] md:min-w-[50%] pr-4 lg:pr-12">
                    <h3 className="text-3xl font-black tracking-tighter">
                      Vesuvius Challenge Team
                    </h3>
                    {team.challenge.map((t, i) => (
                      <Link link={t} key={i} />
                    ))}
                  </div>
                  <div className="flex-1 flex-col lg:gap-0 gap-2 mt-8 min-w-[100%] md:min-w-[50%] pr-4 lg:pr-12">
                    <h3 className="text-3xl font-black tracking-tighter">
                      EduceLab Team
                    </h3>
                    {team.educe.map((t, i) => (
                      <Link link={t} key={i} />
                    ))}
                  </div>
                  <div className="flex-1 flex-col lg:gap-0 gap-2 mt-8 min-w-[100%] md:min-w-[50%] pr-4 lg:pr-12">
                    <h3 className="text-3xl font-black tracking-tighter">
                      Advisors & Alumni
                    </h3>
                    {team.alumni.map((t, i) => (
                      <Link link={t} key={i} />
                    ))}
                  </div>
                  <div className="flex-1 flex-col lg:gap-0 gap-2 mt-8 min-w-[100%] md:min-w-[50%] pr-4 lg:pr-12">
                    <h3 className="text-3xl font-black tracking-tighter text=[--ifm-color-primary]">
                      Papyrology Team
                    </h3>
                    {team.papyrology.map((t, i) => (
                      <Link link={t} key={i} />
                    ))}
                  </div>
                  <div className="flex-1 flex-col lg:gap-0 gap-2 mt-8 min-w-[100%] md:min-w-[50%] pr-4 lg:pr-12">
                    <h3 className="text-3xl font-black tracking-tighter text=[--ifm-color-primary]">
                      Papyrology Advisors
                    </h3>
                    {team.papyrologyAdvisors.map((t, i) => (
                      <Link link={t} key={i} />
                    ))}
                  </div>
                </div>
                &nbsp;
                <br />
                <Link
                  link={{
                    name: "Villa dei Papiri art by Rocío Espín",
                    href: "https://www.artstation.com/rocioespin",
                  }}
                />
              </div>
              <div className="py-10">
                <h1 className="mb-16 text-4xl md:text-7xl font-black leading-none tracking-tighter  mix-blend-exclusion">
                  Partners
                </h1>
                <div className="flex lg:flex-row flex-col lg:gap-12 gap-6 lg:items-center items-start">
                  {partners.map((p, i) => (
                    <a key={i} href={p.href}>
                      <img src={p.icon} className={`h-${i === 1 ? 28 : 12}`} />
                    </a>
                  ))}
                </div>
                <div className="flex flex-wrap">
                  <div className="flex-1 flex-col lg:gap-0 gap-2 mt-8 min-w-[100%] md:min-w-[50%] pr-4 lg:pr-12">
                    <h3
                      className="text-3xl font-black tracking-tighter"
                      id="educelab-funders"
                    >
                      EduceLab funders
                    </h3>
                    {educelabFunders.map((t, i) => (
                      <Link link={t} key={i} />
                    ))}
                  </div>
                </div>
              </div>
            </div>
          </section>
        </div>
      </div>
    </>
  );
}
