import React from "react";

/**
 * Emits a Schema.org JSON-LD block. Rendered directly in the page BODY (not via
 * <Head>) on purpose: Docusaurus's <Head>/react-helmet strips <script> children
 * from the server-rendered HTML, so head-injected JSON-LD is invisible to
 * non-JS crawlers. JSON-LD is valid anywhere in the document, and a body
 * <script type="application/ld+json"> is server-rendered into the static HTML.
 *
 * Usage in an .md/.mdx page:
 *   import JsonLd from '@site/src/components/JsonLd';
 *   <JsonLd data={{ "@context": "https://schema.org", "@type": "Dataset", ... }} />
 */
export default function JsonLd({ data }) {
  if (!data) return null;
  return (
    <script
      type="application/ld+json"
      dangerouslySetInnerHTML={{ __html: JSON.stringify(data) }}
    />
  );
}
