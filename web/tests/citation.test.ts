import { describe, expect, test } from "vitest";
import { classifyCitation } from "../components/citation";

describe("classifyCitation", () => {
  test("classifies an http(s) URL as a followable link", () => {
    expect(classifyCitation("https://arxiv.org/abs/2201.11903")).toEqual({
      kind: "url",
      href: "https://arxiv.org/abs/2201.11903",
    });
    expect(classifyCitation("http://example.com/paper")).toEqual({
      kind: "url",
      href: "http://example.com/paper",
    });
  });

  test("classifies a graph community label distinctly", () => {
    expect(classifyCitation("community-3")).toEqual({ kind: "community", href: null });
  });

  test("classifies a plain filename as a file, with no href", () => {
    expect(classifyCitation("bm25.pdf")).toEqual({ kind: "file", href: null });
  });

  test("classifies a nested file path as a file, with no href", () => {
    expect(classifyCitation("data/documents/paper.pdf")).toEqual({ kind: "file", href: null });
  });

  test("does not treat a non-http scheme as a followable link", () => {
    expect(classifyCitation("javascript:alert(1)")).toEqual({ kind: "file", href: null });
    expect(classifyCitation("ftp://example.com/x")).toEqual({ kind: "file", href: null });
  });
});
