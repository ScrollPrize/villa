/** @type {import('tailwindcss').Config} */
module.exports = {
  corePlugins: {
    preflight: false, // disable Tailwind's reset
  },
  content: [
    "./docs/**/*.{md,mdx}",
    "./src/**/*.{js,jsx}"
  ],
  darkMode: ["class", '[data-theme="dark"]'], // hooks into docusaurus' dark mode settings
  theme: {
    extend: {
      colors: {
        // Token bridge (spec §1). Implementation agents use these class names
        // (bg-surface, border-line, text-accent…) — never new hex arbitrary values.
        bg: "var(--vc-bg)",
        surface: "var(--vc-surface)",
        raised: "var(--vc-raised)",
        line: "var(--vc-line)",
        dim: "var(--vc-text-dim)",
        faint: "var(--vc-text-faint)",
        accent: "var(--vc-accent)",
        gold: "var(--vc-gold)", // legacy atlas gold; keeps existing gold utilities working
      },
    },
  },
  plugins: [],
  future: {
    // hoverOnlyWhenSupported: true,
  },
};
