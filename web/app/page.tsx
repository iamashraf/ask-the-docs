"use client";

import Link from "next/link";
import { useState } from "react";
import { ChatMessage, type Message } from "../components/ChatMessage";

export default function Page() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);

  const canSend = input.trim().length > 0 && !loading;

  async function sendMessage() {
    const query = input.trim();
    if (!query) return;

    setMessages((prev) => [...prev, { role: "user", text: query }]);
    setInput("");
    setLoading(true);

    try {
      const res = await fetch("/api/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query }),
      });
      const data = await res.json();

      if (!res.ok) {
        setMessages((prev) => [...prev, { role: "error", text: data.error }]);
      } else {
        setMessages((prev) => [
          ...prev,
          { role: "assistant", text: data.answer, citations: data.citations },
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
        <h1>Research Assistant</h1>
        <p>Grounded answers, cited sources.</p>
        <nav>
          <Link href="/ingest">Add documents</Link>
        </nav>
      </header>
      <div className="message-log">
        {messages.map((message, i) => (
          <ChatMessage key={i} message={message} />
        ))}
      </div>
      <form
        className="composer"
        onSubmit={(e) => {
          e.preventDefault();
          if (canSend) sendMessage();
        }}
      >
        <input
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
