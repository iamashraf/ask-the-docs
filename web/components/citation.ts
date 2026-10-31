export type CitationKind = "url" | "community" | "file";

export type ClassifiedCitation = {
  kind: CitationKind;
  href: string | null;
};

const COMMUNITY_PATTERN = /^community-\d+$/i;

// Only an http(s) citation is ever safe/meaningful to link out to -- any
// other scheme (javascript:, data:, ftp:, ...) is treated as a plain,
// non-interactive label instead of risking an unsafe href.
export function classifyCitation(citation: string): ClassifiedCitation {
  if (/^https?:\/\//i.test(citation)) {
    return { kind: "url", href: citation };
  }
  if (COMMUNITY_PATTERN.test(citation)) {
    return { kind: "community", href: null };
  }
  return { kind: "file", href: null };
}
