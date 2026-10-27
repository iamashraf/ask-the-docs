"use client";

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
    } finally {
      setLoading(false);
    }
  }

  return (
    <main>
      <h1>Research Assistant</h1>
      <div className="message-log">
        {messages.map((message, i) => (
          <ChatMessage key={i} message={message} />
        ))}
      </div>
      <input
        type="text"
        value={input}
        onChange={(e) => setInput(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === "Enter" && canSend) sendMessage();
        }}
      />
      <button type="button" disabled={!canSend} onClick={sendMessage}>
        Send
      </button>
    </main>
  );
}
