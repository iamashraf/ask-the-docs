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

  test("renders a URL citation as a real, safe followable link", () => {
    render(
      <ChatMessage
        message={{
          role: "assistant",
          text: "Answer text",
          citations: ["https://arxiv.org/abs/2201.11903"],
        }}
      />,
    );
    const link = screen.getByRole("link", { name: "https://arxiv.org/abs/2201.11903" });
    expect(link).toHaveAttribute("href", "https://arxiv.org/abs/2201.11903");
    expect(link).toHaveAttribute("target", "_blank");
    expect(link).toHaveAttribute("rel", "noopener noreferrer");
  });

  test("renders markdown in an assistant answer (lists, emphasis, code)", () => {
    const { container } = render(
      <ChatMessage
        message={{
          role: "assistant",
          text: "Here is **bold** text:\n\n- one\n- two\n\n`inline code`",
        }}
      />,
    );
    expect(container.querySelector("strong")).toHaveTextContent("bold");
    expect(container.querySelectorAll("li")).toHaveLength(2);
    expect(container.querySelector("code")).toHaveTextContent("inline code");
  });

  test("renders an inline [n] citation marker as a styled, non-interactive chip", () => {
    const { container } = render(
      <ChatMessage
        message={{ role: "assistant", text: "BM25 is a ranking function [1]." }}
      />,
    );
    const marker = container.querySelector(".citation-marker");
    expect(marker).not.toBeNull();
    expect(marker?.tagName).toBe("SUP");
    expect(marker).toHaveTextContent("[1]");
  });

  test("renders a non-URL citation (file or community) as a plain label, not a link", () => {
    render(
      <ChatMessage
        message={{ role: "assistant", text: "Answer text", citations: ["bm25.pdf"] }}
      />,
    );
    expect(screen.queryByRole("link")).not.toBeInTheDocument();
    expect(screen.getByText("bm25.pdf")).toBeInTheDocument();
  });
});
