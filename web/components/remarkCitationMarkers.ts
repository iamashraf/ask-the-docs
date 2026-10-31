// remark plugin: splits an inline "[n]" citation marker out of plain text
// nodes into its own node, tagged with data.hName so remark-rehype (inside
// react-markdown) renders it as a real <sup class="citation-marker"> --
// no `components` override or extra dependency needed. Only walks nodes
// that have a `children` array, so leaves (e.g. inlineCode's `value`) are
// never touched, and a literal "[1]" typed inside a code span stays literal.

type MdastNode = {
  type: string;
  value?: string;
  children?: MdastNode[];
  data?: unknown;
};

const MARKER_PATTERN = /\[\d+\]/;
const MARKER_PATTERN_GLOBAL = /\[\d+\]/g;

export function remarkCitationMarkers() {
  return (tree: MdastNode) => {
    visit(tree);
  };
}

function visit(node: MdastNode): void {
  if (!node.children) return;

  const next: MdastNode[] = [];
  for (const child of node.children) {
    if (child.type === "text" && typeof child.value === "string" && MARKER_PATTERN.test(child.value)) {
      next.push(...splitTextNode(child.value));
    } else {
      visit(child);
      next.push(child);
    }
  }
  node.children = next;
}

function splitTextNode(value: string): MdastNode[] {
  const parts: MdastNode[] = [];
  let lastIndex = 0;
  let match: RegExpExecArray | null;

  MARKER_PATTERN_GLOBAL.lastIndex = 0;
  while ((match = MARKER_PATTERN_GLOBAL.exec(value))) {
    if (match.index > lastIndex) {
      parts.push({ type: "text", value: value.slice(lastIndex, match.index) });
    }
    parts.push({
      type: "citationMarker",
      data: { hName: "sup", hProperties: { className: ["citation-marker"] } },
      children: [{ type: "text", value: match[0] }],
    });
    lastIndex = match.index + match[0].length;
  }
  if (lastIndex < value.length) {
    parts.push({ type: "text", value: value.slice(lastIndex) });
  }
  return parts;
}
