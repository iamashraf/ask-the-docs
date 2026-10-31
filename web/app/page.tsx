"use client";

import Link from "next/link";
import { useEffect, useRef, useState } from "react";
import { ChatMessage, type Message } from "../components/ChatMessage";

// The backend caps a history turn's content at 4000 chars (Turn.content in
// router.py); truncate below that with margin so a long grounded answer
// doesn't make the *next* request fail backend validation.
const HISTORY_CONTENT_LIMIT = 3000;
const TRUNCATION_MARKER = "... [truncated]";

function truncateForHistory(text: string): string {
  return text.length > HISTORY_CONTENT_LIMIT
    ? text.slice(0, HISTORY_CONTENT_LIMIT) + TRUNCATION_MARKER
    : text;
}

export default function Page() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [elapsedSeconds, setElapsedSeconds] = useState(0);
  const inputRef = useRef<HTMLInputElement>(null);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    inputRef.current?.focus();
  }, []);

  // No backend signal exists for "which stage is this request in" --
  // /v1/ask is one synchronous call -- so this is a deliberately honest
  // indeterminate indicator (an elapsed-time counter) rather than a fake
  // staged "Retrieving... / Generating..." sequence.
  useEffect(() => {
    if (!loading) {
      setElapsedSeconds(0);
      return;
    }
    const id = setInterval(() => setElapsedSeconds((s) => s + 1), 1000);
    return () => clearInterval(id);
  }, [loading]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView();
  }, [messages, loading]);

  const canSend = input.trim().length > 0 && !loading;

  async function sendMessage() {
    const query = input.trim();
    if (!query) return;

    const history = messages
      .filter((message) => message.role !== "error")
      .map((message) => ({ role: message.role, content: truncateForHistory(message.text) }));

    setMessages((prev) => [...prev, { role: "user", text: query }]);
    setInput("");
    setLoading(true);

    try {
      const res = await fetch("/api/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query, history }),
      });
      const data = await res.json();

      if (!res.ok) {
        setMessages((prev) => [...prev, { role: "error", text: data.error }]);
      } else {
        setMessages((prev) => [
          ...prev,
          { role: "assistant", text: data.answer, citations: data.citations, kind: data.kind },
        ]);
      }
    } catch {
      // The browser couldn't reach /api/ask at all (e.g. the web
      // container itself is down) -- /api/ask's own route handler already
      // degrades a backend-side failure to a normal JSON error response,
      // so this only covers the client-side fetch failing outright.
      setMessages((prev) => [
        ...prev,
        { role: "error", text: "Couldn't reach the server — is it running?" },
      ]);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="app">
      <header className="masthead">
        <h1>AskTheDocs</h1>
        <p>Grounded answers, cited sources.</p>
        <nav>
          <Link href="/ingest">Add documents</Link>
        </nav>
      </header>
      <div className="message-log">
        {messages.length === 0 && !loading && (
          <p className="empty-state">Ask a question about the indexed documents to get started.</p>
        )}
        {messages.map((message, i) => (
          <ChatMessage key={i} message={message} />
        ))}
        {loading && (
          <div className="message message--assistant message--thinking">
            <p role="status">Thinking…</p>
            <p className="message__elapsed" aria-hidden="true">
              {elapsedSeconds}s
            </p>
          </div>
        )}
        <div ref={bottomRef} />
      </div>
      <form
        className="composer"
        onSubmit={(e) => {
          e.preventDefault();
          if (canSend) sendMessage();
        }}
      >
        <input
          ref={inputRef}
          type="text"
          aria-label="Question"
          placeholder="Ask a question about the indexed documents…"
          value={input}
          onChange={(e) => setInput(e.target.value)}
        />
        <button type="submit" disabled={!canSend}>
          Send
        </button>
      </form>
    </div>
  );
}
