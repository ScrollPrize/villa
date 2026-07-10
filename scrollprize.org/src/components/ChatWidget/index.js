import React, {
  Suspense,
  lazy,
  useCallback,
  useEffect,
  useRef,
  useState,
} from "react";
import useDocusaurusContext from "@docusaurus/useDocusaurusContext";
import useIsBrowser from "@docusaurus/useIsBrowser";
import ChatAvatar from "./ChatAvatar";

// The panel (renderer + streaming logic) stays out of the initial JS bundle;
// it loads the first time the trigger is pressed.
const ChatPanel = lazy(() =>
  import(/* webpackChunkName: "chat-panel" */ "./ChatPanel")
);

const STORAGE_KEY = "vc-chat-conversation-v1";
const MAX_HISTORY = 8;
const MAX_CONTENT_CHARS = 1500;

// Payload per the API contract: roles alternating, last message the user's,
// capped at the last MAX_HISTORY entries. Walks backward so an unanswered
// user message (failed send) collapses into the retry instead of doubling up.
function buildPayload(msgs) {
  const out = [];
  for (let i = msgs.length - 1; i >= 0 && out.length < MAX_HISTORY; i--) {
    const m = msgs[i];
    if (!m.content) continue;
    if (out.length && out[out.length - 1].role === m.role) continue;
    out.push({ role: m.role, content: m.content.slice(0, MAX_CONTENT_CHARS) });
  }
  out.reverse();
  // Capping an alternating conversation at an even count leaves an
  // assistant-first array, which some providers reject — trim to user-first.
  if (out.length && out[0].role === "assistant") out.shift();
  return out;
}

function readStoredConversation() {
  try {
    const raw = window.sessionStorage.getItem(STORAGE_KEY);
    if (!raw) return null;
    const saved = JSON.parse(raw);
    if (!Array.isArray(saved)) return null;
    const cleaned = saved
      .filter(
        (m) =>
          m &&
          (m.role === "user" || m.role === "assistant") &&
          typeof m.content === "string" &&
          m.content
      )
      .map((m) => ({
        role: m.role,
        content: m.content,
        interrupted: !!m.interrupted,
      }));
    return cleaned.length ? cleaned : null;
  } catch (e) {
    return null; // corrupt or blocked storage — start fresh
  }
}

export default function ChatWidget() {
  const { siteConfig } = useDocusaurusContext();
  const endpoint =
    (siteConfig.customFields && siteConfig.customFields.chatEndpoint) || "";
  const isBrowser = useIsBrowser();

  const [open, setOpen] = useState(false);
  const [panelRequested, setPanelRequested] = useState(false);
  const [messages, setMessages] = useState([]);
  const [status, setStatus] = useState("idle"); // idle | waiting | streaming
  const [error, setError] = useState(null); // null | { kind: "rate" | "generic" }
  const [announcement, setAnnouncement] = useState("");

  const triggerRef = useRef(null);
  const abortRef = useRef(null);
  const busyRef = useRef(false);
  const hydratedRef = useRef(false);
  const messagesRef = useRef(messages);
  messagesRef.current = messages;

  // Mirror the conversation to sessionStorage so a hard reload keeps it.
  // Hydrate in an effect (never during render) — the site is SSR'd.
  useEffect(() => {
    const saved = readStoredConversation();
    if (saved) setMessages(saved);
    hydratedRef.current = true;
  }, []);

  useEffect(() => {
    if (!hydratedRef.current) return;
    try {
      const done = messages
        .filter((m) => m.content)
        .map((m) => ({
          role: m.role,
          content: m.content,
          interrupted: !!m.interrupted,
        }));
      if (done.length) {
        window.sessionStorage.setItem(STORAGE_KEY, JSON.stringify(done));
      } else {
        window.sessionStorage.removeItem(STORAGE_KEY);
      }
    } catch (e) {
      // best effort — private browsing / quota
    }
  }, [messages]);

  const send = useCallback(
    async (text) => {
      const content = (text || "").trim().slice(0, MAX_CONTENT_CHARS);
      if (!content || busyRef.current) return;
      busyRef.current = true;
      setError(null);

      const base = messagesRef.current.filter((m) => m.content);
      const payload = buildPayload([...base, { role: "user", content }]);
      setMessages([
        ...base,
        { role: "user", content },
        { role: "assistant", content: "", pending: true },
      ]);
      setStatus("waiting");

      const controller = new AbortController();
      abortRef.current = controller;
      let received = ""; // accumulated answer (for the aria-live announcement)

      const appendChunk = (chunk) => {
        received += chunk;
        setMessages((prev) => {
          const last = prev[prev.length - 1];
          if (!last || last.role !== "assistant") return prev;
          return [
            ...prev.slice(0, -1),
            { ...last, content: last.content + chunk },
          ];
        });
      };

      // Settle the pending assistant message: drop it if empty, otherwise
      // mark it done (and interrupted when the stream broke mid-answer).
      const settle = (interrupted) => {
        setMessages((prev) => {
          const last = prev[prev.length - 1];
          if (!last || last.role !== "assistant" || !last.pending) return prev;
          if (!last.content) return prev.slice(0, -1);
          return [
            ...prev.slice(0, -1),
            { role: "assistant", content: last.content, interrupted },
          ];
        });
      };

      try {
        const res = await fetch(endpoint, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ messages: payload }),
          signal: controller.signal,
        });
        if (!res.ok || !res.body) {
          settle(false);
          setError({ kind: res.status === 429 ? "rate" : "generic" });
          return;
        }
        const reader = res.body.getReader();
        const decoder = new TextDecoder();
        for (;;) {
          const { done, value } = await reader.read();
          if (done) break;
          const chunk = decoder.decode(value, { stream: true });
          if (!chunk) continue;
          if (!received) setStatus("streaming");
          appendChunk(chunk);
        }
        const tail = decoder.decode();
        if (tail) appendChunk(tail);
        settle(false);
        // Announce ONCE per completed message (never per token).
        if (received) setAnnouncement(received);
      } catch (err) {
        if (received) {
          // Stop button or a dropped connection mid-answer: keep the partial
          // text with an unobtrusive marker.
          settle(true);
        } else {
          settle(false);
          if (!(err && err.name === "AbortError")) {
            setError({ kind: "generic" });
          }
        }
      } finally {
        busyRef.current = false;
        abortRef.current = null;
        setStatus("idle");
      }
    },
    [endpoint]
  );

  const stop = useCallback(() => {
    if (abortRef.current) abortRef.current.abort();
  }, []);

  const clear = useCallback(() => {
    if (abortRef.current) abortRef.current.abort();
    setMessages([]);
    setError(null);
    setAnnouncement("");
    try {
      window.sessionStorage.removeItem(STORAGE_KEY);
    } catch (e) {
      // best effort
    }
  }, []);

  const openPanel = useCallback(() => {
    setPanelRequested(true);
    setOpen(true);
  }, []);

  // Escape and the × both land here; focus returns to the trigger.
  const closePanel = useCallback(() => {
    setOpen(false);
    if (triggerRef.current) triggerRef.current.focus();
  }, []);

  // No endpoint configured — the widget doesn't exist.
  if (!endpoint) return null;

  return (
    <>
      <button
        type="button"
        ref={triggerRef}
        className="vc-chat-trigger"
        aria-label="Virtual Philodemus — site assistant"
        aria-haspopup="dialog"
        aria-expanded={open}
        onClick={open ? closePanel : openPanel}
      >
        <ChatAvatar size={26} className="vc-chat-avatar--bare" />
        <span className="vc-chat-trigger__long">Ask Philodemus</span>
        <span className="vc-chat-trigger__short">Ask</span>
      </button>
      {isBrowser && panelRequested && (
        <Suspense fallback={null}>
          <ChatPanel
            open={open}
            messages={messages}
            status={status}
            error={error}
            onSend={send}
            onStop={stop}
            onClear={clear}
            onClose={closePanel}
          />
        </Suspense>
      )}
      <div className="vc-chat-sr" aria-live="polite" aria-atomic="true">
        {announcement}
      </div>
    </>
  );
}
