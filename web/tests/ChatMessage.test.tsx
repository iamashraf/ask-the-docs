import { render, screen } from "@testing-library/react";
import { describe, expect, test } from "vitest";
import { ChatMessage } from "../components/ChatMessage";

describe("ChatMessage", () => {
  test("renders assistant text and citations", () => {
    render(
      <ChatMessage
        message={{
          role: "assistant",
          text: "Answer text",
          citations: ["paper.pdf", "community-3"],
        }}
      />,
    );
    expect(screen.getByText("Answer text")).toBeInTheDocument();
    expect(screen.getByText("paper.pdf")).toBeInTheDocument();
    expect(screen.getByText("community-3")).toBeInTheDocument();
  });

  test("renders a user message with no citations", () => {
    render(<ChatMessage message={{ role: "user", text: "What is BM25?" }} />);
    expect(screen.getByText("What is BM25?")).toBeInTheDocument();
  });

  test("renders an error message distinctly from an assistant message", () => {
    const { container } = render(
      <ChatMessage message={{ role: "error", text: "Couldn't reach the server" }} />,
    );
    expect(screen.getByText(/Couldn't reach the server/)).toBeInTheDocument();
    expect(container.querySelector(".message--error")).not.toBeNull();
    expect(container.querySelector(".message--assistant")).toBeNull();
  });

  test("an error message is visibly labeled, not just differently classed", () => {
    render(<ChatMessage message={{ role: "error", text: "Couldn't reach the server" }} />);
    // role="alert" is both an accessibility signal and a real, testable
    // guarantee that an error renders differently from a plain message --
    // a CSS class alone isn't enough (a reviewer specifically flagged this).
    expect(screen.getByRole("alert")).toBeInTheDocument();
    expect(screen.getByText(/^Error:/)).toBeInTheDocument();
  });
});
