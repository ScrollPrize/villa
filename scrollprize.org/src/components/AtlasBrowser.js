import React, { useEffect, useRef, useState } from 'react';
import BrowserOnly from '@docusaurus/BrowserOnly';

const ATLAS_CSS_URL = '/atlas/assets/index-D41dWO2V.css';
const ATLAS_JS_URL = '/atlas/assets/index-CNv2_hr7.js';
const ATLAS_READY_EVENT = 'atlas-container-ready';
const ATLAS_GLOBAL_FLAG = '__atlasBrowserLoaded';

function AtlasBrowserInner() {
  const containerRef = useRef(null);
  const [loaded, setLoaded] = useState(false);
  const [error, setError] = useState(null);

  // react-helmet-async removes data-theme from <html> during SPA page transitions
  // (its cleanup runs in a rAF after React commit, before new page's Helmet re-applies).
  // Since this site forces dark mode (disableSwitch: true), use a MutationObserver to
  // immediately restore it whenever it gets removed.
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
    // Also restore immediately in case it's already missing
    if (!document.documentElement.getAttribute('data-theme')) {
      document.documentElement.setAttribute('data-theme', 'dark');
    }
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    const existingLink = document.querySelector(`link[href="${ATLAS_CSS_URL}"]`);
    const existingScript = document.querySelector(`script[src="${ATLAS_JS_URL}"]`);
    let link = existingLink;
    let script = existingScript;

    const dispatchContainerReady = () => {
      console.log('Atlas assets ready');
      if (script) {
        script.dataset.atlasBrowserLoaded = 'true';
      }
      window[ATLAS_GLOBAL_FLAG] = true;
      setLoaded(true);
      window.dispatchEvent(new CustomEvent(ATLAS_READY_EVENT));
    };

    if (window[ATLAS_GLOBAL_FLAG] && existingLink && existingScript) {
      dispatchContainerReady();
      return;
    }

    if (!link) {
      link = document.createElement('link');
      link.rel = 'stylesheet';
      link.href = ATLAS_CSS_URL;
      link.onload = () => console.log('CSS loaded');
      link.onerror = () => setError('Failed to load CSS');
      link.setAttribute('data-atlas-browser', '1');
      document.head.appendChild(link);
    }

    const onScriptLoad = () => dispatchContainerReady();
    const onScriptError = () => setError('Failed to load JS');

    if (!script) {
      script = document.createElement('script');
      script.type = 'module';
      script.src = ATLAS_JS_URL;
      script.setAttribute('data-atlas-browser', '1');
      script.addEventListener('load', onScriptLoad, { once: true });
      script.addEventListener('error', onScriptError, { once: true });
      document.body.appendChild(script);
      return;
    }

    // If script already loaded from a previous mount, mount immediately.
    if (script.dataset.atlasBrowserLoaded === 'true' || script.readyState === 'complete') {
      dispatchContainerReady();
      return;
    }

    // Otherwise wait for the in-flight load to finish.
    script.addEventListener('load', onScriptLoad, { once: true });
    script.addEventListener('error', onScriptError, { once: true });

    // Don't cleanup on unmount - keep the Atlas loaded for SPA navigation
  }, []);

  return (
    <div>
      {error && <div style={{ padding: '20px', color: 'red' }}>Error: {error}</div>}
      {!loaded && !error && <div style={{ padding: '20px' }}>Loading atlas...</div>}
      <div
        id="atlas-root"
        ref={containerRef}
        style={{
          width: '100%',
          minHeight: 'calc(100vh - 60px)',
        }}
      />
    </div>
  );
}

export default function AtlasBrowser() {
  return (
    <BrowserOnly fallback={<div>Loading...</div>}>
      {() => <AtlasBrowserInner />}
    </BrowserOnly>
  );
}
