import React from "react";
import Link from "@docusaurus/Link";

// Markdown-lite renderer per the widget contract: ONLY [text](url) links,
// **bold**, `inline code`, and "- " bullet lists — built as React elements,
// never dangerouslySetInnerHTML. Anything else renders as plain text.

const SITE_ORIGIN = /^https?:\/\/(www\.)?scrollprize\.org(?=\/|#|\?|$)/i;

function renderLink(label, url, key) {
  const children = renderInline(label, key);
  // Internal links client-side navigate; the panel stays open (state persists).
  if (url.startsWith("/") || url.startsWith("#")) {
    return (
      <Link key={key} to={url}>
        {children}
      </Link>
    );
  }
  if (SITE_ORIGIN.test(url)) {
    return (
      <Link key={key} to={url.replace(SITE_ORIGIN, "") || "/"}>
        {children}
      </Link>
    );
  }
  if (/^https?:\/\//i.test(url)) {
    return (
      <a key={key} href={url} target="_blank" rel="noopener noreferrer">
        {children}
      </a>
    );
  }
  if (/^mailto:/i.test(url)) {
    return (
      <a key={key} href={url}>
        {children}
      </a>
    );
  }
  // Unknown scheme (javascript: etc.) — never a link.
  return <span key={key}>{label}</span>;
}

function renderInline(text, keyPrefix) {
  // Fresh regex per call: renderInline recurses (links/code inside **bold**),
  // and a shared global regex would have its lastIndex clobbered mid-scan.
  const inline = /\*\*([^*]+)\*\*|`([^`]+)`|\[([^\]]+)\]\(([^)\s]+)\)/g;
  const nodes = [];
  let last = 0;
  let k = 0;
  let match;
  while ((match = inline.exec(text)) !== null) {
    if (match.index > last) nodes.push(text.slice(last, match.index));
    const key = `${keyPrefix}.${k}`;
    if (match[1] !== undefined) {
      nodes.push(<strong key={key}>{renderInline(match[1], key)}</strong>);
    } else if (match[2] !== undefined) {
      nodes.push(<code key={key}>{match[2]}</code>);
    } else {
      nodes.push(renderLink(match[3], match[4], key));
    }
    last = inline.lastIndex;
    k += 1;
  }
  if (last < text.length) nodes.push(text.slice(last));
  return nodes;
}

export default function MarkdownLite({ text }) {
  const blocks = [];
  const lines = String(text).split(/\r?\n/);
  let i = 0;
  let key = 0;
  while (i < lines.length) {
    if (lines[i].trim() === "") {
      i += 1;
      continue;
    }
    if (/^\s*-\s+/.test(lines[i])) {
      const items = [];
      while (i < lines.length && /^\s*-\s+/.test(lines[i])) {
        items.push(lines[i].replace(/^\s*-\s+/, ""));
        i += 1;
      }
      blocks.push(
        <ul key={`ul${key}`}>
          {items.map((item, j) => (
            <li key={j}>{renderInline(item, `ul${key}.${j}`)}</li>
          ))}
        </ul>
      );
    } else {
      const para = [];
      while (
        i < lines.length &&
        lines[i].trim() !== "" &&
        !/^\s*-\s+/.test(lines[i])
      ) {
        para.push(lines[i]);
        i += 1;
      }
      const parts = [];
      para.forEach((line, j) => {
        if (j) parts.push(<br key={`br${j}`} />);
        parts.push(...renderInline(line, `p${key}.${j}`));
      });
      blocks.push(<p key={`p${key}`}>{parts}</p>);
    }
    key += 1;
  }
  return <>{blocks}</>;
}
