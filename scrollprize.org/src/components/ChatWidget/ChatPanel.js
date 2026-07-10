import React, { useCallback, useEffect, useRef, useState } from "react";
import MarkdownLite from "./MarkdownLite";

const EXAMPLES = [
  "How do I get started?",
  "What prizes are open?",
  "Can I use the scroll data commercially?",
];

const MOBILE_QUERY = "(max-width: 996px)";
const INPUT_MAX_HEIGHT = 104; // 4 lines x 20px + 24px padding

export default function ChatPanel({
  open,
  messages,
  status,
  error,
  onSend,
  onStop,
  onClear,
  onClose,
}) {
  const [draft, setDraft] = useState("");
  const [isMobile, setIsMobile] = useState(
    () => window.matchMedia(MOBILE_QUERY).matches
  );
  const panelRef = useRef(null);
  const inputRef = useRef(null);
  const listRef = useRef(null);
  const stickRef = useRef(true); // stay glued to the bottom unless scrolled up

  const busy = status !== "idle";

  useEffect(() => {
    const mql = window.matchMedia(MOBILE_QUERY);
    const onChange = () => setIsMobile(mql.matches);
    mql.addEventListener("change", onChange);
    return () => mql.removeEventListener("change", onChange);
  }, []);

  useEffect(() => {
    if (!open) return;
    // Autofocusing the input on touch devices pops the soft keyboard over the
    // empty state's suggestion chips — there, move focus to the panel itself
    // (keyboard stays down, dialog focus still moves in for a11y).
    const fine = window.matchMedia("(hover: hover) and (pointer: fine)").matches;
    if (fine && inputRef.current) inputRef.current.focus();
    else if (panelRef.current) panelRef.current.focus();
  }, [open]);

  // Full-screen sheet: the page behind must not scroll.
  useEffect(() => {
    if (!(open && isMobile)) return undefined;
    const prev = document.documentElement.style.overflow;
    document.documentElement.style.overflow = "hidden";
    return () => {
      document.documentElement.style.overflow = prev;
    };
  }, [open, isMobile]);

  // Textarea autogrow, 1–4 lines.
  useEffect(() => {
    const el = inputRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, INPUT_MAX_HEIGHT)}px`;
  }, [draft]);

  const onListScroll = useCallback(() => {
    const el = listRef.current;
    if (!el) return;
    stickRef.current = el.scrollHeight - el.scrollTop - el.clientHeight < 48;
  }, []);

  useEffect(() => {
    const el = listRef.current;
    if (el && stickRef.current) el.scrollTop = el.scrollHeight;
  }, [messages, error, open]);

  const submit = useCallback(() => {
    const text = draft.trim();
    if (!text || busy) return;
    setDraft("");
    onSend(text);
  }, [draft, busy, onSend]);

  // Escape must work document-wide while the panel is open: after an in-panel
  // citation link client-side-navigates, Docusaurus moves focus to <body>, so
  // a keydown handler on the panel (which only sees events bubbling from its
  // descendants) would never fire again.
  useEffect(() => {
    if (!open) return undefined;
    const onDocKeyDown = (e) => {
      if (e.key === "Escape") onClose();
    };
    document.addEventListener("keydown", onDocKeyDown);
    return () => document.removeEventListener("keydown", onDocKeyDown);
  }, [open, onClose]);

  const onKeyDown = (e) => {
    if (e.key === "Escape") {
      // stopPropagation keeps the event from reaching the document-level
      // listener (no double-close) and from page-level handlers underneath —
      // so this branch must close directly.
      e.stopPropagation();
      onClose();
      return;
    }
    // Focus trap while the mobile sheet is up (desktop card is non-modal).
    if (e.key !== "Tab" || !isMobile) return;
    const root = panelRef.current;
    if (!root) return;
    const focusables = Array.from(
      root.querySelectorAll("button:not([disabled]), a[href], textarea")
    );
    if (!focusables.length) return;
    const first = focusables[0];
    const last = focusables[focusables.length - 1];
    if (e.shiftKey && document.activeElement === first) {
      e.preventDefault();
      last.focus();
    } else if (!e.shiftKey && document.activeElement === last) {
      e.preventDefault();
      first.focus();
    }
  };

  const onInputKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      submit();
    }
  };

  return (
    <div
      ref={panelRef}
      className="vc-chat-panel"
      data-open={open ? "true" : "false"}
      tabIndex={-1}
      role="dialog"
      aria-modal={isMobile ? "true" : undefined}
      aria-label="Ask the Scrolls"
      onKeyDown={onKeyDown}
    >
      <header className="vc-chat-head">
        <span className="vc-chat-head__title">Ask the Scrolls</span>
        {messages.length > 0 && (
          <button
            type="button"
            className="vc-chat-head__clear"
            onClick={onClear}
          >
            Clear
          </button>
        )}
        <button
          type="button"
          className="vc-chat-head__close"
          aria-label="Close chat"
          onClick={onClose}
        >
          <svg width="14" height="14" viewBox="0 0 14 14" aria-hidden="true">
            <path
              d="M2 2l10 10M12 2L2 12"
              stroke="currentColor"
              strokeWidth="1.4"
              strokeLinecap="round"
            />
          </svg>
        </button>
      </header>

      <div className="vc-chat-list" ref={listRef} onScroll={onListScroll}>
        {messages.length === 0 && !error && (
          <div className="vc-chat-empty">
            <div className="vc-chat-empty__mark" aria-hidden="true">
              <svg width="28" height="28" viewBox="0 0 16 16" fill="none">
                <path
                  d="M2.75 3.25h10.5a1 1 0 0 1 1 1v6a1 1 0 0 1-1 1H8.4l-3.15 2.9v-2.9H2.75a1 1 0 0 1-1-1v-6a1 1 0 0 1 1-1Z"
                  stroke="currentColor"
                  strokeWidth="1"
                  strokeLinejoin="round"
                />
              </svg>
            </div>
            <p className="vc-chat-empty__hint">
              Ask about the Vesuvius Challenge — answers come from this
              site&rsquo;s content.
            </p>
            {EXAMPLES.map((q) => (
              <button
                key={q}
                type="button"
                className="vc-chat-chip"
                onClick={() => onSend(q)}
              >
                {q}
              </button>
            ))}
          </div>
        )}
        {messages.map((m, i) =>
          m.role === "user" ? (
            <div key={i} className="vc-chat-msg vc-chat-msg--user">
              {m.content}
            </div>
          ) : (
            <div key={i} className="vc-chat-msg vc-chat-msg--assistant">
              {m.content ? (
                <MarkdownLite text={m.content} />
              ) : (
                <span className="vc-chat-dots" aria-hidden="true">
                  <i />
                  <i />
                  <i />
                </span>
              )}
              {m.interrupted && (
                <span className="vc-chat-interrupted">— interrupted</span>
              )}
            </div>
          )
        )}
        {error && (
          <div className="vc-chat-error" role="alert">
            {error.kind === "rate" ? (
              "You're sending messages quickly — try again in a moment."
            ) : (
              <>
                Something went wrong reaching the assistant. Try again, or{" "}
                <a
                  href="https://discord.gg/uTfNwwecCQ"
                  target="_blank"
                  rel="noopener noreferrer"
                >
                  ask on Discord
                </a>
                .
              </>
            )}
          </div>
        )}
      </div>

      <form
        className="vc-chat-inputrow"
        onSubmit={(e) => {
          e.preventDefault();
          submit();
        }}
      >
        <textarea
          ref={inputRef}
          className="vc-chat-input"
          rows={1}
          maxLength={1500}
          placeholder="Ask a question…"
          aria-label="Your question"
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          onKeyDown={onInputKeyDown}
        />
        {busy ? (
          <button
            type="button"
            className="vc-chat-stop"
            aria-label="Stop generating"
            onClick={onStop}
          >
            Stop
          </button>
        ) : (
          <button
            type="submit"
            className="vc-chat-send"
            aria-label="Send message"
            disabled={!draft.trim()}
          >
            <svg
              width="16"
              height="16"
              viewBox="0 0 16 16"
              fill="none"
              aria-hidden="true"
            >
              <path
                d="M8 13V3M3.5 7.5 8 3l4.5 4.5"
                stroke="currentColor"
                strokeWidth="1.8"
                strokeLinecap="round"
                strokeLinejoin="round"
              />
            </svg>
          </button>
        )}
      </form>
      <p className="vc-chat-foot">
        Answers are AI-generated from site content.
      </p>
    </div>
  );
}
