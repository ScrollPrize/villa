// useAtlasCard.js
//
// Hook used by each scroll/fragment card to hook its `.view` element into the
// shared renderer. Returns a ref to attach to the card's `.view` div; on mount
// it registers (id, element, mesh) with the AtlasRendererProvider, and on
// unmount it unregisters (disposing the GPU resources for that card).

import { useContext, useEffect, useRef } from 'react';
import { AtlasRendererContext } from './AtlasRendererProvider';

/**
 * @param {string} id   - unique card id (used as the registry key).
 * @param {object|null} mesh - manifest entry { file, kind }, or null/undefined
 *                             if this card has no 3D model.
 * @returns {React.RefObject} viewRef - attach to the card's `.view` element.
 */
export default function useAtlasCard(id, mesh) {
  const viewRef = useRef(null);
  const api = useContext(AtlasRendererContext);

  useEffect(() => {
    // Only register cards that actually have a mesh, an attached element, and a
    // live provider context.
    if (!mesh || !viewRef.current || !api) return undefined;
    const el = viewRef.current;
    api.register(id, el, mesh);
    return () => api.unregister(id);
    // `api` is a stable ref-held object, so it is intentionally omitted from
    // deps; re-register only when the id or mesh changes.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [id, mesh]);

  return viewRef;
}
