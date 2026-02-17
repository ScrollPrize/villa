import React, { useEffect, useRef, useState } from 'react';
import BrowserOnly from '@docusaurus/BrowserOnly';

function AtlasBrowserInner() {
  const containerRef = useRef(null);
  const [loaded, setLoaded] = useState(false);
  const [error, setError] = useState(null);
  const scriptRef = useRef(null);
  const linkRef = useRef(null);

  useEffect(() => {
    console.log('AtlasBrowser mounting...');
    console.log('Container element:', containerRef.current);

    // Don't load if already loaded (prevents double loading on remount)
    if (scriptRef.current || linkRef.current) {
      console.log('Atlas already loaded, dispatching container-ready event...');
      setLoaded(true);
      // Dispatch event to tell Atlas to mount
      window.dispatchEvent(new CustomEvent('atlas-container-ready'));
      return;
    }

    // Dynamically load the atlas CSS
    const link = document.createElement('link');
    link.rel = 'stylesheet';
    link.href = '/atlas/assets/index-Ca7AkimD.css';
    link.onload = () => console.log('CSS loaded');
    link.onerror = () => setError('Failed to load CSS');
    document.head.appendChild(link);
    linkRef.current = link;

    // Dynamically load the atlas JS
    const script = document.createElement('script');
    script.type = 'module';
    script.src = '/atlas/assets/index-BPwaJqtD.js';
    script.onload = () => {
      console.log('JS loaded');
      setLoaded(true);
      // Dispatch event after script loads
      window.dispatchEvent(new CustomEvent('atlas-container-ready'));
    };
    script.onerror = () => setError('Failed to load JS');
    document.body.appendChild(script);
    scriptRef.current = script;

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
