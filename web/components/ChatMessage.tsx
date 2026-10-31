import Link from "next/link";
import ReactMarkdown from "react-markdown";
import { classifyCitation } from "./citation";
import { remarkCitationMarkers } from "./remarkCitationMarkers";

export type Message = {
  role: "user" | "assistant" | "error";
  text: string;
  citations?: string[];
  kind?: "answer" | "refusal" | "chitchat" | "meta";
};

const CITATION_ICON: Record<ReturnType<typeof classifyCitation>["kind"], string> = {
  url: "🔗",
  community: "🕸️",
  file: "📄",
};

export const REFUSAL_MESSAGE =
  "I don't have enough information in the available documents to answer that.";

export function ChatMessage({ message }: { message: Message }) {
  const isError = message.role === "error";
  const isAssistant = message.role === "assistant";
  const isRefusal = isAssistant && message.kind === "refusal";
  const isNotGrounded = isAssistant && (message.kind === "chitchat" || message.kind === "meta");
  const classes = ["message", `message--${message.role}`];
  if (isRefusal) classes.push("message--refusal");

  return (
    <div className={classes.join(" ")} role={isError ? "alert" : undefined}>
      {isError ? (
        <p>{`Error: ${message.text}`}</p>
      ) : isRefusal ? (
        <>
          <p>{message.text}</p>
          <p className="message__refusal-hint">
            Nothing in the indexed documents matched.{" "}
            <Link href="/ingest">Add documents</Link> to expand what AskTheDocs can answer.
          </p>
        </>
      ) : isAssistant ? (
        <>
          {isNotGrounded && <p className="message__not-grounded">Not from your documents</p>}
          <div className="message__markdown">
            <ReactMarkdown remarkPlugins={[remarkCitationMarkers]}>{message.text}</ReactMarkdown>
          </div>
        </>
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
