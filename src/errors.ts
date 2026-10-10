/** Errors only the audio library raises, over the core's hierarchy. */

import { YnONNXError, registerErrors } from "@youni/onnxweb-core/core";

export class YnONNXAudioError extends YnONNXError {
  static override readonly code: string = "YnONNXAudioError";
}

/** Word timings could not be aligned to the audio. */
export class AlignmentError extends YnONNXAudioError {
  static override readonly code: string = "AlignmentError";
}

/** Speakers could not be separated. */
export class DiarizationError extends YnONNXAudioError {
  static override readonly code: string = "DiarizationError";
}

/** The spoken language could not be told. */
export class LanguageDetectionError extends YnONNXAudioError {
  static override readonly code: string = "LanguageDetectionError";
}

/** The section and wave scheduler stopped making progress. */
export class SchedulingError extends YnONNXAudioError {
  static override readonly code: string = "SchedulingError";
}

/** A speaker print that cannot be made or used: too little speech, or an unknown name. */
export class EnrollmentError extends YnONNXAudioError {
  static override readonly code: string = "EnrollmentError";
}

/**
 * A streaming session broke its protocol (a revision after a final one) or cannot run in the mode
 * asked.
 */
export class StreamingError extends YnONNXAudioError {
  static override readonly code: string = "StreamingError";
}

registerErrors(
  YnONNXAudioError,
  StreamingError,
  EnrollmentError,
  AlignmentError,
  DiarizationError,
  LanguageDetectionError,
  SchedulingError,
);
