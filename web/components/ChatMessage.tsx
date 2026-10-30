import Link from "next/link";
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

// The API has no explicit "this was a refusal" flag -- /v1/ask always
// returns this exact string (see ask_the_docs.config.REFUSAL_MESSAGE) with
// an empty citations list, whether retrieval found nothing or the
// generated answer failed the citation guardrail twice. Matching on the
// literal string is a real coupling to that backend constant (the e2e
// suite already duplicates it the same way), but there's no other signal
// to key off without a contract change.
export const REFUSAL_MESSAGE =
  "I don't have enough information in the available documents to answer that.";

export function ChatMessage({ message }: { message: Message }) {
  const isError = message.role === "error";
  const isAssistant = message.role === "assistant";
  const isRefusal = isAssistant && message.text === REFUSAL_MESSAGE;
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
