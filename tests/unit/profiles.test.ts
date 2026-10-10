import { describe, expect, it } from "vitest";

import { catalogueKey, profileFor } from "../../src/whisper/profiles.js";

describe("decoding profiles", () => {
  it("reads the catalogue key out of a manifest model key", () => {
    expect(catalogueKey("whisper-small_grain-int4-b64")).toBe("small");
    expect(catalogueKey("whisper-large-v3-turbo_algo-int4-hqq")).toBe("large-v3-turbo");
    expect(catalogueKey("distil-large-v3.5_plain-int8")).toBe("distil-large-v3.5");
    expect(catalogueKey("whisper-tiny.en")).toBe("tiny.en");
  });

  it("repairs the Distil checkpoints trained for chunked decoding", () => {
    expect(profileFor("distil-small.en_sym-int4-asym")).toEqual({
      conditionOnPreviousText: false,
      noRepeatNgramSize: 3,
    });
  });

  it("gives an unmeasured checkpoint nothing rather than failing", () => {
    expect(profileFor("whisper-tiny_plain-int4")).toEqual({});
    expect(profileFor("someone/else_int4")).toEqual({});
  });

  it("keeps the three roles apart where they differ", () => {
    expect(profileFor("whisper-large-v3-turbo", "default", "en")).toEqual({});
    expect(profileFor("whisper-large-v3-turbo", "speed", "en")).toEqual({
      conditionOnPreviousText: false,
    });
  });

  it("hands back a copy, so a caller cannot edit the table", () => {
    const first = profileFor("whisper-large-v3", "default", "en");
    first.conditionOnPreviousText = true;
    expect(profileFor("whisper-large-v3", "default", "en").conditionOnPreviousText).toBe(false);
  });
});

describe("profiles depend on the language", () => {
  it("gives a multilingual model its profile on English audio", () => {
    expect(profileFor("whisper-large-v3", "default", "en")).toEqual({
      conditionOnPreviousText: false,
    });
  });

  it("gives a multilingual model the stock settings in French", () => {
    expect(profileFor("whisper-large-v3", "default", "fr")).toEqual({});
  });

  it("does not assume English when the language is unknown", () => {
    expect(profileFor("whisper-large-v3")).toEqual({});
    expect(profileFor("whisper-large-v3", "default", null)).toEqual({});
  });

  it("ignores the language for an English-only checkpoint", () => {
    const expected = profileFor("distil-small.en");
    expect(profileFor("distil-small.en", "default", "fr")).toEqual(expected);
    expect(profileFor("distil-small.en", "default", "en")).toEqual(expected);
  });
});
