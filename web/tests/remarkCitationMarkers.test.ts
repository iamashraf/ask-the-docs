import { describe, expect, test } from "vitest";
import { remarkCitationMarkers } from "../components/remarkCitationMarkers";

// mdast nodes are plain objects -- exercising the transformer directly
// (no React, no unified pipeline) keeps this fast and focused on the one
// thing it does: splitting "[n]" out of text nodes into its own node type.
function textNode(value: string) {
  return { type: "text", value };
}

describe("remarkCitationMarkers", () => {
  test("splits a trailing citation marker out of a text node", () => {
    const tree = {
      type: "root",
      children: [{ type: "paragraph", children: [textNode("BM25 is a ranking function [1].")] }],
    };

    remarkCitationMarkers()(tree as never);

    const paragraphChildren = (tree.children[0] as { children: unknown[] }).children;
    expect(paragraphChildren).toEqual([
      textNode("BM25 is a ranking function "),
      {
        type: "citationMarker",
        data: { hName: "sup", hProperties: { className: ["citation-marker"] } },
        children: [textNode("[1]")],
      },
      textNode("."),
    ]);
  });

  test("splits multiple markers in the same text node", () => {
    const tree = {
      type: "root",
      children: [{ type: "paragraph", children: [textNode("See [1] and [2].")] }],
    };

    remarkCitationMarkers()(tree as never);

    const paragraphChildren = (tree.children[0] as { children: unknown[] }).children;
    expect(paragraphChildren.map((n) => (n as { type: string }).type)).toEqual([
      "text",
      "citationMarker",
      "text",
      "citationMarker",
      "text",
    ]);
  });

  test("leaves text with no citation marker untouched", () => {
    const tree = {
      type: "root",
      children: [{ type: "paragraph", children: [textNode("No markers here.")] }],
    };

    remarkCitationMarkers()(tree as never);

    expect((tree.children[0] as { children: unknown[] }).children).toEqual([
      textNode("No markers here."),
    ]);
  });

  test("does not touch inline code (no children array), so a literal [1] in code stays literal", () => {
    const tree = {
      type: "root",
      children: [
        { type: "paragraph", children: [{ type: "inlineCode", value: "[1]" }] },
      ],
    };

    remarkCitationMarkers()(tree as never);

    expect((tree.children[0] as { children: unknown[] }).children).toEqual([
      { type: "inlineCode", value: "[1]" },
    ]);
  });

  test("recurses into nested formatting (e.g. a marker inside bold text)", () => {
    const tree = {
      type: "root",
      children: [
        {
          type: "paragraph",
          children: [{ type: "strong", children: [textNode("bold [3] text")] }],
        },
      ],
    };

    remarkCitationMarkers()(tree as never);

    const strongChildren = (
      (tree.children[0] as { children: unknown[] }).children[0] as { children: unknown[] }
    ).children;
    expect(strongChildren).toEqual([
      textNode("bold "),
      {
        type: "citationMarker",
        data: { hName: "sup", hProperties: { className: ["citation-marker"] } },
        children: [textNode("[3]")],
      },
      textNode(" text"),
    ]);
  });
});
