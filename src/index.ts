/** Whisper in the browser. */

export * from "./core.js";

export { DEFAULT_VAD, SILERO_REPO, VAD, segmentsFromProbabilities } from "./audio/vad.js";
export type { SpeechSegment, VADConfig, VADLoadOptions } from "./audio/vad.js";
export {
  LARGE_MODEL_PARAMS_M,
  WHISPER_PARAMS_M,
  applyPadding,
  clipToSpeech,
  groupSpeech,
  restoreTime,
  shouldUseVad,
  speechCoverage,
} from "./audio/vadPolicy.js";
export type { SpeechMap, VADMode } from "./audio/vadPolicy.js";
export { TimestampRules } from "./whisper/rules.js";
export type { TimestampRulesOptions, TimestampVocabulary } from "./whisper/rules.js";

export {
  AudioTranscriber,
  LANGUAGE_PROBES,
  MOONSHINE_BOS,
  MOONSHINE_EOS,
  MOONSHINE_FAMILIES,
  MAX_TARGET_POSITIONS,
  WHISPER_FAMILIES,
  RETRY_LOGPROB,
  dominantLanguage,
  mergePortions,
  portionAt,
  loadTarget,
  meanLogprob,
  paramsOf,
  probePoints,
} from "./whisper/transcriber.js";
export type {
  AudioAnalysisResult,
  LanguageDetectionResult,
  TranscribeOptions,
  TranscribeProgress,
  TranscriptionResult,
  TranscriberOptions,
  TranscriberStreamOptions,
  LanguagePortion,
} from "./whisper/transcriber.js";

export {
  DEFAULT_VARIANTS,
  EXTRA_VARIANTS,
  PUBLISHER,
  WEBGPU_INCOMPATIBLE,
  WEB_VARIANTS,
  asRoles,
  MOONSHINE_REPOS,
  isPublished,
  moonshineSource,
  knownVariants,
  namedVariant,
  publishedSource,
  repoFor,
  variantFor,
  variantForRole,
  widthOf,
} from "./whisper/published.js";
export type { Half, HalfPrecision, PublishedSource, PublishedSourceOptions } from "./whisper/published.js";

export {
  CLIP_LEVEL,
  FRAME_MS,
  MIN_SILENCE_S,
  SILENCE_BELOW_SPEECH_DB,
  TELEPHONE_HIGH_HZ,
  percentile,
  profileAudio,
} from "./audio/profile.js";
export type { AudioProfile } from "./audio/profile.js";
export {
  DECODING_ROLE,
  DEFAULT_LEVEL,
  LEVELS,
  MEMORY_MARGIN,
  NOISY_SNR_DB,
  PRECISION_LADDER,
  WER_BUDGET_POINTS,
  browserHardware,
  candidates,
  checkLevel,
  choose,
  explain,
  fits,
  isMultilingualModel,
  label,
  ladder,
  levelFor,
  modelMemoryMb,
  parseAuto,
  pick,
  resolveSettings,
  retryRung,
  scalarFallback,
  werCells,
} from "./whisper/resolver.js";
export { WEB_MEASUREMENTS } from "./whisper/autoTables.js";
export type {
  Candidate,
  Decision,
  Device,
  Hardware,
  Level,
  Measurements,
  ModelMeasurements,
  Resolution,
  ResolveOptions,
} from "./whisper/resolver.js";
export {
  DEFAULT_CTC_MODEL,
  FRAMES_PER_SECOND,
  ForcedAligner,
  ctcVocabulary,
  logSoftmaxRows,
  normalizeForCtc,
  normalizeSignal,
  toTargets,
  viterbiAlign,
  wordsFromFrames,
} from "./alignment/ctc.js";
export type { CTCVocabulary, ForcedAlignerOptions } from "./alignment/ctc.js";
export { MEDIAN_FILTER_WIDTH, SECONDS_PER_FRAME, dtwPath, groupIntoWords, medianFilter, tokenSpans } from "./alignment/dtw.js";
export {
  APPEND_PUNCTUATIONS,
  PREPEND_PUNCTUATIONS,
  mergePunctuation,
  proportionalWords,
  splitWordsIntoSegments,
} from "./alignment/punctuation.js";
export type { Word } from "./alignment/punctuation.js";
export {
  CHUNK_SECONDS,
  CHUNK_STRIDE,
  DEFAULT_EMBEDDING_MODEL,
  DEFAULT_SEGMENTATION_MODEL,
  ENROLLED_MATCH_THRESHOLD,
  MIN_CHUNK_SECONDS,
  matchEnrolled,
  SpeakerDiarizer,
  assignSpeakers,
  chunkSpeech,
  overlapSeconds,
  runsToSpans,
  speakerDurations,
  speakerInfoOf,
  displayName,
  toTurns,
  MERGE_GAP_SECONDS,
  SEGMENTATION_WINDOW_S,
  SILENCE_CLASS,
} from "./diarization/diarizer.js";
export type { DiarizationConfig, DiarizerOptions, SpeakerInfo, SpeakerTurn } from "./diarization/diarizer.js";
export { agglomerative, cosineSimilarityMatrix, kmeans, relabelByFirstAppearance, spectral, symmetricEigen } from "./diarization/clustering.js";
export { NUM_MEL_BINS, fbank } from "./diarization/fbank.js";
export { ANALYSES, AVAILABLE_ANALYSES, TIMELINE_SCHEMA_VERSION, Timeline, checkAnalyses, emptyTimeline } from "./timeline.js";
export type { Analysis, Annotation, AnnotationKind, Granularity, TimelineData } from "./timeline.js";
export { RENDER_FORMATS, renderCsv, renderMarkdown, renderResult } from "./render.js";
export type { RenderFormat, RenderableResult } from "./render.js";
export { MODEL_LICENSES, PLAN_SCHEMA_VERSION, allowed, buildPlan, explainEntries, explainPlan, licenseOf } from "./plan.js";
export type { ExecutionPlan, LicenseCategory, LicensePolicy, ModelLicense, NetworkPolicy, Origin, PlanEntry, PlanInputs, PlanStep, StepName } from "./plan.js";
export { GAP, WaveJournal } from "./whisper/journal.js";
export { DEFAULT_MAX_WINDOWS, ENCODER_POSITIONS, maxWindows, sectionBudget, windowBytes } from "./whisper/batching.js";
export type { WaveBudget } from "./whisper/batching.js";

export { decodeAudio, resampleTo16k } from "./audio/load.js";

export { WorkerTranscriber } from "./worker/client.js";
export { AUDIO_ONLY_ANALYSES, AudioAnalyzer, mergeAnalysis, writeDetections } from "./analysis/analyzer.js";
export type { AudioAnalysisOptions, AudioAnalyzerOptions, AudioOnlyAnalysis } from "./analysis/analyzer.js";
export type { WorkerTranscriberOptions } from "./worker/client.js";
export { FAST_WINDOW_S, LARGE_STEP_S, MAX_WINDOW_S, SETTLE_MARGIN_S, StreamSession } from "./streaming.js";
export type { Producer, StreamMode, StreamOptions, StreamPlan, StreamResult, StreamSegment, StreamUpdate, StreamingTranscriber } from "./streaming.js";
export { AudioSetTagger, DEFAULT_EMOTION_MODEL, DEFAULT_EVENTS_MODEL, EMOTIONS, EmotionClassifier, audioSetKind, canonicalEmotion, canonicalEvent } from "./analysis/classifiers.js";
export type { ClassifierOptions, Detection, Emotion } from "./analysis/classifiers.js";
