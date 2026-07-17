import React from "react";
import ChatWidget from "@site/src/components/ChatWidget";

// Root persists across client-side route navigation, so the chat widget (and
// its conversation state) survives page changes. Children render untouched.
export default function Root({ children }) {
  return (
    <>
      {children}
      <ChatWidget />
    </>
  );
}
