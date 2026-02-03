import React, { useEffect, useRef, useState } from 'react';
import BrowserOnly from '@docusaurus/BrowserOnly';

function AtlasBrowserInner() {
  const containerRef = useRef(null);
  const [loaded, setLoaded] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    console.log('AtlasBrowser mounting...');
    console.log('Container element:', containerRef.current);

    // Dynamically load the atlas CSS
    const link = document.createElement('link');
    link.rel = 'stylesheet';
    link.href = '/atlas/assets/index-DFHlRhaq.css';
    link.onload = () => console.log('CSS loaded');
    link.onerror = () => setError('Failed to load CSS');
    document.head.appendChild(link);

    // Dynamically load the atlas JS
    const script = document.createElement('script');
    script.type = 'module';
    script.src = '/atlas/assets/index-Dns3DuQW.js';
    script.onload = () => {
      console.log('JS loaded');
      setLoaded(true);
    };
    script.onerror = () => setError('Failed to load JS');
    document.body.appendChild(script);

    return () => {
      // Cleanup on unmount
      if (document.head.contains(link)) {
        document.head.removeChild(link);
      }
      if (document.body.contains(script)) {
        document.body.removeChild(script);
      }
    };
  }, []);

  return (
    <div>
      {error && <div style={{ padding: '20px', color: 'red' }}>Error: {error}</div>}
      {!loaded && !error && <div style={{ padding: '20px' }}>Loading atlas...</div>}
      <div
        id="atlas-root"
        className="dark"
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
