/**
 * Lines Whisper invents: the subtitle credits of its training captions, and lines with no word in
 * them.
 */

/** Longest segment, in words, that can be a credit line. */
export const MAX_CREDIT_WORDS = 15;

const W = "[\\p{L}\\p{N}_]";

/** The credit lines, on lowercased text without accents. */
export const CREDIT_PATTERNS: readonly RegExp[] = [
  `sous[- ]?titr${W}*\\b.*\\b(radio[- ]?canada|amara|mfp|st ?'? ?\\d+|par )`,
  "\\bamara\\.?org\\b",
  "\\bcommunaute d ?amara\\b",
  `\\bsubtitl${W}* (by|provided)\\b`,
  `\\buntertitel${W}* (im auftrag|von|der|des|durch)\\b`,
  "\\bsubtitulos? (realizados|por|hechos)\\b",
  "\\bsottotitoli (creati|a cura|realizzati)\\b",
  "\\blegendas (pela|por)\\b",
  "\\bondertiteld door\\b",
].map((pattern) => new RegExp(pattern, "u"));

/** Lowercase, accents removed, curly apostrophe straightened, spaces collapsed. */
function plain(text: string): string {
  const stripped = text.toLowerCase().normalize("NFKD").replace(/\p{M}/gu, "");
  return stripped.replace(/’/g, "'").replace(/\s+/g, " ").trim();
}

/** Is this segment one of the subtitle credits Whisper invents? */
export function isCreditLine(text: string): boolean {
  const line = plain(text);
  if (!line || line.split(" ").length > MAX_CREDIT_WORDS) return false;
  return CREDIT_PATTERNS.some((pattern) => pattern.test(line));
}

/**
 * A whole segment that describes a sound instead of saying words: `*cris*`, `[Musique]`,
 * `(Applaudissements)`.
 */
const DESCRIPTION = /^\s*(\*[^*]*\*|\[[^\]]*\]|\([^)]*\))\s*$/u;

/**
 * A segment with no word in it: only punctuation (`...`, `- -`, `♪`), or a description of a sound
 * between `*`, `[]` or `()`. An empty line is not one (there is nothing to drop).
 */
export function isNonSpeechLine(text: string): boolean {
  if (!/[\p{L}\p{N}]/u.test(text)) return text.trim().length > 0;
  return DESCRIPTION.test(text);
}

/** A segment that is never speech: a subtitle credit or a line with no word. */
export function isInventedLine(text: string, _language?: string | null): boolean {
  return isCreditLine(text) || isNonSpeechLine(text);
}
