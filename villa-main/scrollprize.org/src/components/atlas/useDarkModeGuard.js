// useDarkModeGuard.js
//
// This site forces dark mode (Docusaurus `colorMode.disableSwitch: true`).
// react-helmet-async removes `data-theme` from <html> during SPA page
// transitions (its cleanup runs in a rAF after React commit, before the new
// page's Helmet re-applies). This hook installs a MutationObserver that
// immediately restores `data-theme="dark"` whenever the attribute is removed,
// so the page never flashes to light mode during client-side navigation.
//
// Extracted verbatim (behavior-wise) from the original AtlasBrowser.js.

import { useEffect } from 'react';

export default function useDarkModeGuard() {
  useEffect(() => {
    const observer = new MutationObserver(() => {
      if (!document.documentElement.getAttribute('data-theme')) {
        document.documentElement.setAttribute('data-theme', 'dark');
      }
    });
    observer.observe(document.documentElement, {
      attributes: true,
      attributeFilter: ['data-theme'],
    });
    // Also restore immediately in case it's already missing on mount.
    if (!document.documentElement.getAttribute('data-theme')) {
      document.documentElement.setAttribute('data-theme', 'dark');
    }
    return () => observer.disconnect();
  }, []);
}
