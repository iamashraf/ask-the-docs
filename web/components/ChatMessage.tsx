export type Message = {
  role: "user" | "assistant" | "error";
  text: string;
  citations?: string[];
};

export function ChatMessage({ message }: { message: Message }) {
  const isError = message.role === "error";
  return (
    <div className={`message message--${message.role}`} role={isError ? "alert" : undefined}>
      <p>{isError ? `Error: ${message.text}` : message.text}</p>
      {message.citations && message.citations.length > 0 && (
        <ul className="message__citations">
          {message.citations.map((citation) => (
            <li key={citation}>{citation}</li>
          ))}
        </ul>
      )}
    </div>
  );
}
