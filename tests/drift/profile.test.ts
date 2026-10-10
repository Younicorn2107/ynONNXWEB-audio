import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

import { describe, expect, it } from "vitest";

import { percentile, profileAudio } from "../../src/audio/profile.js";

const f = JSON.parse(
  readFileSync(join(dirname(fileURLToPath(import.meta.url)), "..", "fixtures", "profile.json"), "utf-8"),
) as Record<string, Record<string, number>>;

/** The generator's signal, sample for sample (scripts/gen_fixtures.py, `profile_signal`). */
function signal(kind: string, seconds = 8): Float32Array {
  const n = seconds * 16000;
  const out = new Float32Array(n);
  let state = 12345;
  const level = kind === "noisy" ? 0.3 : 0.01;
  for (let i = 0; i < n; i += 1) {
    state = Number((1103515245n * BigInt(state) + 12345n) % 2147483648n);
    const noise = state / 2147483648 - 0.5;
    const t = i / 16000;
    const envelope = 0.5 + 0.5 * Math.sin(2 * Math.PI * 3 * t);
    let voice = 0.3 * Math.sin(2 * Math.PI * 180 * t) + 0.2 * Math.sin(2 * Math.PI * 1200 * t);
    if (kind === "bright") voice += 0.1 * Math.sin(2 * Math.PI * 5000 * t);
    let x = voice * envelope + level * noise;
    if (kind === "gaps" && ((t >= 2 && t < 3.5) || (t >= 5 && t < 6.2))) x = 0.0001 * noise;
    if (kind === "clipped") x = Math.max(-1, Math.min(1, 4 * x));
    out[i] = x;
  }
  return out;
}

const FIELDS: Record<string, string> = {
  duration_s: "durationS",
  snr_estimate_db: "snrEstimateDb",
  energy_above_telephone: "energyAboveTelephone",
  bandwidth_hz: "bandwidthHz",
  speech_level_dbfs: "speechLevelDbfs",
  peak_dbfs: "peakDbfs",
  clipping_ratio: "clippingRatio",
  silence_ratio: "silenceRatio",
  longest_silence_s: "longestSilenceS",
};

describe("audio profile, against ynlibs", () => {
  for (const kind of Object.keys(f)) {
    it(`measures '${kind}' as Python does`, () => {
      const got = profileAudio(signal(kind)) as unknown as Record<string, number>;
      for (const [python, field] of Object.entries(FIELDS)) {
        expect(got[field], `${kind}.${field}`).toBeCloseTo(f[kind]![python]!, 1);
      }
    });
  }

  it("interpolates percentiles like numpy", () => {
    expect(percentile([1, 2, 3, 4], 50)).toBe(2.5);
    expect(percentile([10, 0, 5], 90)).toBe(9);
  });

  it("returns a neutral profile for a signal too short to measure", () => {
    const short = profileAudio(new Float32Array(100));
    expect(short.snrEstimateDb).toBe(0);
    expect(short.bandwidthHz).toBe(8000);
  });
});
