/** The decoding settings measured for each checkpoint. */

export type DecodingProfile = "default" | "speed" | "quality";

export interface ProfileSettings {
  conditionOnPreviousText?: boolean;
  noRepeatNgramSize?: number;
}

/**
 * English-only checkpoints: the profile applies whatever language is announced, since there is only
 * one.
 */
const PROFILES: Readonly<
  Record<string, Readonly<Record<DecodingProfile, ProfileSettings>>>
> = {
  "small.en": {
    default: {},
    speed: { conditionOnPreviousText: false },
    quality: {},
  },
  "medium.en": {
    default: {},

    speed: { conditionOnPreviousText: false },
    quality: {},
  },

  "distil-small.en": {
    default: { conditionOnPreviousText: false, noRepeatNgramSize: 3 },
    speed: { conditionOnPreviousText: false, noRepeatNgramSize: 3 },
    quality: { conditionOnPreviousText: false, noRepeatNgramSize: 3 },
  },

  "distil-medium.en": {
    default: { conditionOnPreviousText: false, noRepeatNgramSize: 3 },
    speed: { conditionOnPreviousText: false, noRepeatNgramSize: 3 },
    quality: { conditionOnPreviousText: false, noRepeatNgramSize: 3 },
  },

  "distil-large-v2": {
    default: { conditionOnPreviousText: false },
    speed: { conditionOnPreviousText: false },
    quality: { conditionOnPreviousText: false },
  },

  "distil-large-v3": {
    default: { conditionOnPreviousText: false },
    speed: { conditionOnPreviousText: false, noRepeatNgramSize: 3 },
    quality: { conditionOnPreviousText: false },
  },

  "distil-large-v3.5": {
    default: { noRepeatNgramSize: 3 },
    speed: { noRepeatNgramSize: 3 },
    quality: { noRepeatNgramSize: 3 },
  },
};

/** Multilingual checkpoints, measured on English: applied to English audio only. */
const ENGLISH_AUDIO_PROFILES: Readonly<
  Record<string, Readonly<Record<DecodingProfile, ProfileSettings>>>
> = {
  base: {
    default: { conditionOnPreviousText: false },
    speed: { conditionOnPreviousText: false },
    quality: { conditionOnPreviousText: false },
  },

  small: {
    default: { conditionOnPreviousText: false },
    speed: {},
    quality: { conditionOnPreviousText: false },
  },

  "large-v1": {
    default: { noRepeatNgramSize: 3 },
    speed: { noRepeatNgramSize: 3 },
    quality: { noRepeatNgramSize: 3 },
  },

  "large-v3": {
    default: { conditionOnPreviousText: false },
    speed: { conditionOnPreviousText: false },
    quality: { conditionOnPreviousText: false },
  },
  "large-v3-turbo": {
    default: {},

    speed: { conditionOnPreviousText: false },
    quality: {},
  },
};

/** `"whisper-small_grain-int4-b64"` → `"small"`. */
export function catalogueKey(modelKey: string): string {
  const name = modelKey.split("_", 1)[0] ?? modelKey;
  return name.startsWith("whisper-") ? name.slice("whisper-".length) : name;
}

/** The measured settings for this checkpoint and this language, or nothing. */
export function profileFor(
  modelKey: string,
  profile: DecodingProfile = "default",
  language: string | null = null,
): ProfileSettings {
  const key = catalogueKey(modelKey);
  const englishOnly = PROFILES[key];
  if (englishOnly) return { ...(englishOnly[profile] ?? {}) };
  if (language === "en") return { ...(ENGLISH_AUDIO_PROFILES[key]?.[profile] ?? {}) };
  return {};
}
