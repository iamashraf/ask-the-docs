export type Message = {
  role: "user" | "assistant" | "error";
  text: string;
  citations?: string[];
};

export function ChatMessage({ message }: { message: Message }) {
  return (
    <div className={`message message--${message.role}`}>
      <p>{message.text}</p>
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
