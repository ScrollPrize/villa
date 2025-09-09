/** @type {import('tailwindcss').Config} */
module.exports = {
  corePlugins: {
    preflight: false, // disable Tailwind's reset
  },
  content: [
    "./docusaurus/*.md",
    "./docs/*.md",
    "./src/**/*.js"
  ],
  darkMode: ["class", '[data-theme="dark"]'], // hooks into docusaurus' dark mode settings
  theme: {
    extend: {},
  },
  plugins: [],
  future: {
    // hoverOnlyWhenSupported: true,
  },
};
