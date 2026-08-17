import React from "react";
import useDocusaurusContext from "@docusaurus/useDocusaurusContext";
import ChatAvatar from "./ChatAvatar";

// In-content invitation to the assistant ("Ask Philodemus"), styled by the
// .vc-callout family in chrome.css. Clicking dispatches `vc:open-chat`, which
// ChatWidget (mounted in theme/Root) listens for: it opens the panel and puts
// `prefill` in the input as a draft — never auto-sent.
export default function ChatCallout({
  prefill = "",
  text = "Content feeling overwhelming? Catch up by asking Philodemus.",
}) {
  const { siteConfig } = useDocusaurusContext();
  const endpoint =
    (siteConfig.customFields && siteConfig.customFields.chatEndpoint) || "";
  // Same rule as the widget itself: no endpoint, no assistant — a callout
  // whose button does nothing would be worse than none.
  if (!endpoint) return null;

  const ask = () => {
    if (typeof window === "undefined") return;
    window.dispatchEvent(
      new CustomEvent("vc:open-chat", { detail: { prefill } })
    );
  };

  return (
    <aside className="vc-callout">
      <ChatAvatar size={40} className="vc-callout__avatar" />
      <p className="vc-callout__text">{text}</p>
      <button type="button" className="vc-callout__btn" onClick={ask}>
        Ask Philodemus
      </button>
    </aside>
  );
}
