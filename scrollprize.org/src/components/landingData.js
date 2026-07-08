// landingData.js — pure data arrays for the landing page (OWNED BY WP1).
// Content (names, amounts, links) is preserved verbatim from Landing.js;
// only presentation lives in Landing.js / landing.css.
//
// NOTE: this array now carries AWARDED prizes only. The open-prize board on
// the landing is sourced at build time from docs/34_prizes.md frontmatter via
// plugins/prizes-data.js (usePluginData("prizes-data") in Landing.js), so the
// landing updates automatically when the prizes page changes.

export const prizes = [
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
];

export const creators = [
  {
    name: "Nat Friedman",
    role: "Instigator, Director & Founding Sponsor",
    image: "/img/landing/nat.webp",
    href: "https://nat.org/",
  },
  {
    name: "Daniel Gross",
    role: "Founding Sponsor",
    image: "/img/landing/daniel.webp",
    href: "https://dcgross.com/",
  },
  {
    name: "Brent Seales",
    role: "Principal Advisor, PhD",
    image: "/img/landing/brent.webp",
    href: "https://educelab.engr.uky.edu/w-brent-seales",
  },
];

export const sponsors = [
  {
    name: "Nat Friedman",
    amount: 2250000,
    href: "https://nat.org/",
    image: "/img/landing/nat.webp",
  },
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
    image: ["/img/landing/Julia DeWahl.webp", "/img/landing/Dan Romero.webp"],
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
    image: "/img/landing/Neil Parikh.jpg",
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

export const projectLead = {
  name: "Giorgio Angelotti",
  title: "Project & Tech Team Lead, PhD",
  href: "https://thegiorgio.org/",
};

export const team = {
  challenge: [
    {
      name: "Sean Johnson",
      title: "Research Assistant",
      href: "https://github.com/bruniss",
    },
    {
      name: "Hendrik Schilling",
      title: "Computer Vision & AI Expert, PhD",
      href: "https://www.linkedin.com/in/dr-hendrik-schilling-a2019418a",
    },
    {
      name: "Paul Henderson",
      title: "Computer Vision & AI Expert, PhD",
      href: "https://www.pmh47.net/",
    },
    {
      name: "Elian Rafael Dal Prá",
      title: "ML Intern",
      href: "https://twitter.com/elianrafaeldp",
    },
    {
      name: "Johannes Rudolph",
      title: "Platform Engineer",
      href: "https://blog.virtual-void.net/",
    },
  ],
  annotation: [
    {
      name: "David Josey",
      title: "Team Lead, PhD",
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
      name: "Sarah Morejohn",
      title: "Annotation Specialist",
      href: "https://www.linkedin.com/in/sarah-morejohn-1140b049/",
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
      name: "Stephen Parsons",
      title: "Project Lead, PhD",
      href: "https://www2.cs.uky.edu/dri/stephen-parsons/",
    },
    {
      name: "Youssef Nader",
      title: "Machine Learning Researcher",
      href: "https://youssefnader.com/",
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
      name: "Forrest McDonald",
      title: "Software Engineer",
      href: "https://www.linkedin.com/in/forrest-mcdonald-a80b9885/",
    },
    {
      name: "Adrionna Fey",
      title: "Annotation Specialist",
      href: "https://twitter.com/Meadowsnax1",
    },
    {
      name: "Konrad Rosenberg",
      title: "Annotation Specialist",
      href: "https://twitter.com/germanicgems",
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
      href: "https://www.linkedin.com/in/sergey-pnev",
    },
    {
      name: "Techjays",
      title: "Annotation Services",
      href: "https://www.techjays.com/",
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
      name: "Federica Nicolardi",
      title:
        "Team Lead and Assistant Professor, University of Naples Federico II",
      href: "https://www.docenti.unina.it/federica.nicolardi",
    },
    {
      name: "Marzia D'Angelo",
      title:
        "Postdoctoral Fellow, University of Naples Federico II",
      href: "https://unina.academia.edu/MDAngelo",
    },
    {
      name: "Kilian Fleischer",
      title: "Research Director and Papyrologist, University of Tübingen",
      href: "https://www.klassphil.uni-wuerzburg.de/team/pd-dr-kilian-fleischer/",
    },
    {
      name: "Alessia Lavorante",
      title:
        "Postdoctoral Fellow, University of Naples Federico II",
      href: "https://unina.academia.edu/AlessiaLavorante",
    },
    {
      name: "Michael McOsker",
      title: "Researcher, University College London",
      href: "https://profiles.ucl.ac.uk/97051-michael-mcosker",
    },
    {
      name: "Maria Chiara Robustelli",
      title:
        "Postdoctoral Fellow, University of Naples Federico II",
      href: "https://unina.academia.edu/mrobustelli",
    },
    {
      name: "Claudio Vergara",
      title:
        "Postdoctoral Fellow, University of Naples Federico II",
      href: "https://unina.academia.edu/ClaudioVergara",
    },
    {
      name: "Rossella Villa",
      title: "Research Assistant, University of Salerno",
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
      name: "Tobias Reinhardt",
      title:
        "Corpus Christi Professor of the Latin Language and Literature, Oxford",
      href: "https://www.classics.ox.ac.uk/people/professor-tobias-reinhardt",
    },
  ],
};

export const partners = [
  {
    name: "EduceLab",
    icon: "/img/landing/educe.svg",
    href: "https://educelab.engr.uky.edu/",
  },
  {
    name: "Institut de France",
    icon: "/img/landing/institute.svg",
    href: "https://www.institutdefrance.fr/en/home/",
    tall: true,
  },
  {
    name: "Biblioteca Nazionale di Napoli",
    icon: "/img/landing/biblioteca.svg",
    href: "https://www.bnnonline.it/",
  },
  {
    name: "Getty",
    icon: "/img/landing/getty.svg",
    href: "https://www.getty.edu/",
  },
  {
    name: "Kaggle",
    icon: "/img/landing/kaggle.svg",
    href: "https://www.kaggle.com/",
  },
];

export const educelabFunders = [
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
