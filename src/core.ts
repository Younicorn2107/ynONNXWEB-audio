/** The isomorphic half: everything that needs no browser. */

export {
  logMel,
  logMelBatch,
  melFilters,
  WHISPER_MEL,
  WINDOW_FRAMES,
  WINDOW_SAMPLES,
  WINDOW_SECONDS,
} from "./audio/mel.js";
export type { MelConfig, MelResult } from "./audio/mel.js";

export {
  WhisperTokens,
  SOT,
  SOT_PREV,
  EOT,
  NO_SPEECH,
  NO_TIMESTAMPS,
  TRANSCRIBE,
  TRANSLATE,
  TIME_PRECISION,
  TIMESTAMP_COUNT,
} from "./whisper/tokens.js";

export { buildPrompt, withoutContext, DEFAULT_MAX_CONTEXT } from "./whisper/prompt.js";
export type { PromptOptions } from "./whisper/prompt.js";
export { catalogueKey, profileFor } from "./whisper/profiles.js";
export type { DecodingProfile, ProfileSettings } from "./whisper/profiles.js";

export {
  TEMPERATURE_SCALE,
  DEFAULT_QUALITY,
  avgLogprob,
  avgLogprobFromSum,
  compressionRatio,
  compressionRatioSync,
  isDegenerate,
  isSilent,
  withTemperatureFallback,
} from "./whisper/quality.js";
export type { QualityConfig, WindowScore } from "./whisper/quality.js";

export {
  SectionScheduler,
  WaveScheduler,
  SectionState,
  MIN_SECTION_S,
  DEFAULT_OVERLAP_S,
} from "./whisper/schedule.js";
export type { AudioInfo, Section, WindowResult } from "./whisper/schedule.js";

export {
  extractSegments,
  mergeSegments,
} from "./whisper/segments.js";
export type { Segment, SegmentsResult } from "./whisper/segments.js";

export {
  YnONNXAudioError,
  AlignmentError,
  EnrollmentError,
  StreamingError,
  DiarizationError,
  LanguageDetectionError,
  SchedulingError,
} from "./errors.js";
