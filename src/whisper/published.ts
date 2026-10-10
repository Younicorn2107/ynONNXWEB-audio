/** Which published quantization serves which checkpoint, and per half. */

import { ConfigurationError } from "@youni/onnxweb-core/core";
import { huggingFaceResolver, type ModelResolver } from "@youni/onnxweb-core";

/** The account that owns the repositories. */
export const PUBLISHER = "Younicorn";

/** The most accurate variant of each width (what a native runtime serves). */
export const DEFAULT_VARIANTS: Readonly<Record<string, Readonly<Record<string, string>>>> = {
  tiny: { fp16: "fp16", int8: "grain-int8-b64", int4: "sym-int4-asym-b32" },
  "tiny.en": { fp16: "fp16", int8: "plain-int8", int4: "grain-int4-b32" },
  base: { fp16: "fp16", int8: "edge-int8-1layer", int4: "zone-int4-attn" },
  "base.en": { fp16: "fp16", int8: "grain-int8-b64", int4: "algo-int4-hqq" },
  small: { fp16: "fp16", int8: "plain-int8", int4: "edge-int4-2layers" },
  "small.en": { fp16: "fp16", int8: "grain-int8-b64", int4: "zone-int4-cross" },
  medium: { fp16: "fp16", int8: "plain-int8", int4: "grain-int4-b64" },
  "medium.en": { fp16: "fp16", int8: "plain-int8", int4: "zone-int4-attn" },
  "large-v1": { fp16: "fp16", int8: "plain-int8", int4: "grain-int4-b64" },
  "large-v2": { fp16: "fp16", int8: "plain-int8", int4: "grain-int4-b64" },
  "large-v3": { fp16: "fp16", int8: "plain-int8", int4: "grain-int4-b64" },
  "large-v3-turbo": { fp16: "fp16", int8: "plain-int8", int4: "algo-int4-hqq" },
  "distil-small.en": { fp16: "fp16", int8: "plain-int8", int4: "sym-int4-asym" },
  "distil-medium.en": { fp16: "fp16", int8: "plain-int8", int4: "sym-int4-asym" },
  "distil-large-v2": { fp16: "fp16", int8: "plain-int8", int4: "zone-int4-attn" },
  "distil-large-v3": { fp16: "fp16", int8: "plain-int8", int4: "grain-int4-b64" },
  "distil-large-v3.5": { fp16: "fp16", int8: "plain-int8", int4: "grain-int4-b64" },
};

/**
 * Where the browser's choice differs: the lightest variant. `base.en` and `large-v3-turbo` take
 * `plain-int4` because their most accurate int4 is HQQ, which WebGPU refuses ({@link
 * WEB_INCOMPATIBLE}).
 */
export const WEB_VARIANTS: Readonly<Record<string, Readonly<Record<string, string>>>> = {
  base: { int8: "grain-int8-b32", int4: "sym-int4-asym" },
  "base.en": { int4: "plain-int4" },
  small: { int4: "grain-int4-b64" },
  "medium.en": { int4: "sym-int4-asym" },
  "large-v3-turbo": { int4: "plain-int4" },
};

/** The other published variants: addressable by name (`small@plain-int4`), nobody's default. */
export const EXTRA_VARIANTS: Readonly<Record<string, readonly string[]>> = {
  tiny: ["plain-int4", "plain-int8"],
  "tiny.en": ["grain-int8-b64", "plain-int4"],
  base: ["plain-int4", "plain-int8"],
  "base.en": ["plain-int8"],
  small: ["plain-int4"],
  "small.en": ["plain-int4", "plain-int8"],
  medium: ["plain-int4", "zone-int4-cross"],
  "medium.en": ["plain-int4"],
  "large-v1": ["plain-int4"],
  "large-v2": ["plain-int4"],
  "large-v3": ["plain-int4"],
  "distil-small.en": ["plain-int4"],
  "distil-medium.en": ["algo-int4-hqq", "plain-int4"],
  "distil-large-v2": ["plain-int4"],
  "distil-large-v3": ["plain-int4"],
  "distil-large-v3.5": ["plain-int4"],
};

/**
 * Variants WebGPU cannot run: `algo-int4-hqq` stores float zero points, and WebGPU's `MatMulNBits`
 * wants packed integer ones (packing them was measured to destroy the model, 1.46 % → 122.68 %
 * WER). The CPU kernel takes them.
 */
export const WEBGPU_INCOMPATIBLE: ReadonlySet<string> = new Set(["algo-int4-hqq"]);

export type Half = "encoder" | "decoder";

/** A precision for the whole model, or one per half. */
export type HalfPrecision = string | { encoder?: string; decoder?: string };

const TABLE_KEY: Readonly<Record<string, string>> = { float16: "fp16" };

/** Whether this checkpoint is published. */
export function isPublished(model: string): boolean {
  return model.split("@")[0]! in DEFAULT_VARIANTS;
}

/** The repository that holds this checkpoint. */
export function repoFor(model: string): string {
  const key = model.split("@")[0]!;
  return `${PUBLISHER}/${key.startsWith("distil-") ? key : `whisper-${key}`}`;
}

/** The folder of a width, `null` when nothing is published at it. */
export function variantFor(model: string, precision: string, { web = true } = {}): string | null {
  const width = TABLE_KEY[precision] ?? precision;
  const named = (web ? WEB_VARIANTS[model]?.[width] : undefined) ?? DEFAULT_VARIANTS[model]?.[width];
  return named ? `onnx/${named}` : null;
}

/**
 * The folder holding one half at one precision. A float32 decoder lives in `fp16` (whose decoder is
 * float32); a float16 decoder and a float32 encoder are not published.
 */
export function variantForRole(model: string, half: Half, precision: string, { web = true } = {}): string | null {
  if (half === "decoder" && precision === "float32") return variantFor(model, "fp16", { web });
  if (half === "decoder" && (precision === "float16" || precision === "fp16")) return null;
  if (half === "encoder" && precision === "float32") return null;
  return variantFor(model, precision, { web });
}

/** Every published variant of a checkpoint. */
export function knownVariants(model: string): string[] {
  return [
    ...new Set([
      ...Object.values(DEFAULT_VARIANTS[model] ?? {}),
      ...Object.values(WEB_VARIANTS[model] ?? {}),
      ...(EXTRA_VARIANTS[model] ?? []),
    ]),
  ].sort();
}

/** The width a variant's name announces: `edge-int4-2layers` → `int4`. */
export function widthOf(variant: string): string {
  if (variant === "fp16") return "float16";
  const match = /int(4|8)/.exec(variant);
  if (!match) throw new ConfigurationError(`'${variant}' names no width`);
  return `int${match[1]}`;
}

/**
 * `"small@grain-int4-b64"` → `["small", "grain-int4-b64"]`, `null` without `@`. A variant not
 * published for that checkpoint is refused rather than silently replaced by the default.
 */
export function namedVariant(spec: string): [model: string, variant: string] | null {
  const at = spec.indexOf("@");
  if (at < 0) return null;
  const model = spec.slice(0, at);
  const variant = spec.slice(at + 1);
  if (!knownVariants(model).includes(variant)) {
    throw new ConfigurationError(
      `'${variant}' is not published for '${model}' (published: ${knownVariants(model).join(", ") || "none"})`,
    );
  }
  return [model, variant];
}

/**
 * `"float16"` is the fp16 variant: float16 encoder, float32 decoder. Any other scalar applies to
 * both.
 */
export function asRoles(precision: HalfPrecision): { encoder: string; decoder: string } {
  if (typeof precision !== "string") {
    return { encoder: precision.encoder ?? "float32", decoder: precision.decoder ?? "float32" };
  }
  if (precision === "float16" || precision === "fp16") return { encoder: "float16", decoder: "float32" };
  return { encoder: precision, decoder: precision };
}

export interface PublishedSourceOptions {
  /** The Hub, or a mirror of it. */
  endpoint?: string;
  revision?: string;
  headers?: Record<string, string>;
}

export interface PublishedSource {
  resolver: ModelResolver;
  /** The precisions to hand the loader, one per half (the folder is found from them). */
  computeType: { encoder: string; decoder: string };
}

/** The resolver and per-half precision that load a published checkpoint. */
export function publishedSource(spec: string, precision: HalfPrecision, options: PublishedSourceOptions = {}): PublishedSource {
  const named = namedVariant(spec);
  const model = named ? named[0] : spec;
  if (!isPublished(model)) throw new ConfigurationError(`'${model}' is not a published checkpoint`);
  const roles = named ? { encoder: widthOf(named[1]), decoder: widthOf(named[1]) } : asRoles(precision);
  if (named && roles.encoder === "float16") roles.decoder = "float32";

  const variants: Record<string, string> = {};
  for (const half of ["encoder", "decoder"] as const) {
    const folder = named ? `onnx/${named[1]}` : variantForRole(model, half, roles[half]);
    if (folder === null) {
      throw new ConfigurationError(
        `'${model}' publishes no ${half} at ${roles[half]}` +
          (half === "encoder" && roles[half] === "float32" ? " (the fp16 variant's encoder is float16)" : ""),
      );
    }
    const already = variants[roles[half]];
    if (already !== undefined && already !== folder) {
      throw new ConfigurationError(`'${model}' at ${roles[half]} maps to two folders (${already}, ${folder})`);
    }
    variants[roles[half]] = folder;
  }
  return {
    resolver: huggingFaceResolver({
      repo: repoFor(model),
      variants,
      manifest: true,
      ...(options.endpoint ? { endpoint: options.endpoint } : {}),
      ...(options.revision ? { revision: options.revision } : {}),
      ...(options.headers ? { headers: options.headers } : {}),
    }),
    computeType: roles,
  };
}

/**
 * Moonshine's checkpoints: onnx-community's exports, described from their graphs (they publish no
 * manifest). `moonshine` alone is the base size.
 */
export const MOONSHINE_REPOS: Readonly<Record<string, string>> = {
  "moonshine-tiny": "onnx-community/moonshine-tiny-ONNX",
  "moonshine-base": "onnx-community/moonshine-base-ONNX",
  moonshine: "onnx-community/moonshine-base-ONNX",
};

/**
 * The resolver of a Moonshine checkpoint, or null for any other name. Float32 by default: at 27 to
 * 61 M parameters it costs nothing, and the published merged float16 decoder returns an outer-scope
 * value from a subgraph, which recent ONNX Runtime refuses to load.
 */
export function moonshineSource(name: string, options: PublishedSourceOptions = {}): ModelResolver | null {
  const repo = MOONSHINE_REPOS[name];
  if (!repo) return null;
  return huggingFaceResolver({
    repo,
    subfolder: "onnx",
    manifest: false,
    ...(options.endpoint ? { endpoint: options.endpoint } : {}),
    ...(options.revision ? { revision: options.revision } : {}),
    ...(options.headers ? { headers: options.headers } : {}),
  });
}
