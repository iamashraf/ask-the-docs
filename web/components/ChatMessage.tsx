import ReactMarkdown from "react-markdown";
import { classifyCitation } from "./citation";
import { remarkCitationMarkers } from "./remarkCitationMarkers";

export type Message = {
  role: "user" | "assistant" | "error";
  text: string;
  citations?: string[];
};

const CITATION_ICON: Record<ReturnType<typeof classifyCitation>["kind"], string> = {
  url: "🔗",
  community: "🕸️",
  file: "📄",
};

export function ChatMessage({ message }: { message: Message }) {
  const isError = message.role === "error";
  const isAssistant = message.role === "assistant";
  return (
    <div className={`message message--${message.role}`} role={isError ? "alert" : undefined}>
      {isError ? (
        <p>{`Error: ${message.text}`}</p>
      ) : isAssistant ? (
        <div className="message__markdown">
          <ReactMarkdown remarkPlugins={[remarkCitationMarkers]}>{message.text}</ReactMarkdown>
        </div>
      ) : (
        <p>{message.text}</p>
      )}
      {message.citations && message.citations.length > 0 && (
        <div className="message__citations-wrap">
          <p className="message__citations-label">Sources retrieved</p>
          <ul className="message__citations">
            {message.citations.map((citation) => {
              const { kind, href } = classifyCitation(citation);
              return (
                <li key={citation} className={`citation citation--${kind}`}>
                  <span className="citation__icon" aria-hidden="true">
                    {CITATION_ICON[kind]}
                  </span>
                  {href ? (
                    <a
                      href={href}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="citation__label"
                    >
                      {citation}
                    </a>
                  ) : (
                    <span className="citation__label">{citation}</span>
                  )}
                </li>
              );
            })}
          </ul>
        </div>
      )}
    </div>
  );
}
