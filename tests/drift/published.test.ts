import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

import { describe, expect, it } from "vitest";

import {
  namedVariant,
  publishedSource,
  repoFor,
  variantFor,
  variantForRole,
} from "../../src/whisper/published.js";
import { loadTarget } from "../../src/whisper/transcriber.js";

const f = JSON.parse(
  readFileSync(join(dirname(fileURLToPath(import.meta.url)), "..", "fixtures", "published.json"), "utf-8"),
) as {
  variant_for: Record<string, Record<string, string | null>>;
  variant_for_role: Record<string, Record<"encoder" | "decoder", Record<string, string | null>>>;
  repo_for: Record<string, string>;
  named: Record<string, [string, string] | null>;
};

describe("published variants, against Python's browser table", () => {
  it("serves the same folder for every width", () => {
    for (const [model, widths] of Object.entries(f.variant_for)) {
      for (const [width, folder] of Object.entries(widths)) expect(variantFor(model, width), `${model} ${width}`).toBe(folder);
    }
  });

  it("serves the same folder for every half", () => {
    for (const [model, halves] of Object.entries(f.variant_for_role)) {
      for (const half of ["encoder", "decoder"] as const) {
        for (const [width, folder] of Object.entries(halves[half])) {
          expect(variantForRole(model, half, width), `${model} ${half} ${width}`).toBe(folder);
        }
      }
    }
  });

  it("names the same repositories", () => {
    for (const [model, repo] of Object.entries(f.repo_for)) expect(repoFor(model)).toBe(repo);
  });

  it("reads named variants the same way", () => {
    for (const [spec, expected] of Object.entries(f.named)) {
      const got = namedVariant(spec);
      expect(got && [got[0], `onnx/${got[1]}`]).toEqual(expected);
    }
  });
});

describe("publishedSource", () => {
  const folder = (source: ReturnType<typeof publishedSource>, precision: string) =>
    source.resolver.assetUrl({ name: "x", computeType: precision }, "g.onnx").url;

  it("takes each half from its own folder", () => {
    const source = publishedSource("small", { encoder: "int8", decoder: "float32" });
    expect(source.computeType).toEqual({ encoder: "int8", decoder: "float32" });
    expect(folder(source, "int8")).toContain("/Younicorn/whisper-small/resolve/main/onnx/plain-int8/g.onnx");
    expect(folder(source, "float32")).toContain("/onnx/fp16/g.onnx");
  });

  it("reads float16 as the fp16 variant: float16 encoder, float32 decoder", () => {
    expect(publishedSource("tiny", "float16").computeType).toEqual({ encoder: "float16", decoder: "float32" });
  });

  it("serves a named variant for both halves", () => {
    const source = publishedSource("small@plain-int4", "int8");
    expect(source.computeType).toEqual({ encoder: "int4", decoder: "int4" });
    expect(folder(source, "int4")).toContain("/onnx/plain-int4/");
  });

  it("refuses what is not published", () => {
    expect(() => publishedSource("small", "float32")).toThrow(/no encoder at float32/);
    expect(() => publishedSource("small", { encoder: "int8", decoder: "float16" })).toThrow(/no decoder at float16/);
    expect(() => publishedSource("small@nope-int4", "int4")).toThrow(/not published/);
    expect(() => publishedSource("tiny-unknown", "int8")).toThrow(/not a published/);
  });

  it("goes through a mirror when told to", () => {
    const source = publishedSource("tiny", "int8", { endpoint: "http://localhost:5179/hf" });
    expect(folder(source, "int8")).toBe("http://localhost:5179/hf/Younicorn/whisper-tiny/resolve/main/onnx/grain-int8-b64/g.onnx");
  });
});

describe("loadTarget", () => {
  const base = { ort: {} as never, Tensor: {} as never };

  it("turns a published name into its repository and a precision per half", () => {
    const target = loadTarget({ ...base, model: "small", computeType: { encoder: "int8", decoder: "int4" } });
    expect(target.model).toEqual({ name: "small", computeType: "int4" });
    expect(target.computeType).toEqual({ encoder: "int8", decoder: "int4" });
    expect(target.resolver).toBeDefined();
  });

  it("loads Moonshine from onnx-community, float32 unless told otherwise", () => {
    const target = loadTarget({ ...base, model: "moonshine-tiny" });
    expect(target.model).toEqual({ name: "onnx-community/moonshine-tiny-ONNX", computeType: "float32" });
    expect((target.resolver as { id: string }).id).toBe("huggingface:onnx-community/moonshine-tiny-ONNX");
    expect(loadTarget({ ...base, model: "moonshine", computeType: "int8" }).model).toEqual({ name: "onnx-community/moonshine-base-ONNX", computeType: "int8" });
  });

  it("leaves a model with its own resolver alone", () => {
    const resolver = { id: "mine", manifestUrls: () => [], assetUrl: () => ({ url: "", filename: "" }) };
    const target = loadTarget({ ...base, model: { name: "small", computeType: "int8" }, resolver });
    expect(target).toEqual({ model: { name: "small", computeType: "int8" }, resolver });
  });
});
