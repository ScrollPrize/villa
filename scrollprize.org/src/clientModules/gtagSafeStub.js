// Dev-only safety net for @docusaurus/plugin-google-gtag.
//
// The gtag plugin injects the Google Analytics script (which defines
// window.gtag) ONLY in production builds. In `docusaurus start` (dev) that
// script is never added, yet the plugin's client-side route tracker still
// calls window.gtag('event', 'page_view') on every in-app navigation — which
// throws "window.gtag is not a function" and pops the dev error overlay on
// each internal link click (e.g. opening a scroll from the data browser).
//
// Define a no-op gtag only when one isn't already present. In production the
// inline gtag bootstrap snippet defines window.gtag synchronously before this
// module runs, so the guard is false and real analytics are never touched —
// this only ever takes effect in development.
if (typeof window !== "undefined" && typeof window.gtag !== "function") {
  window.gtag = function gtag() {};
}
