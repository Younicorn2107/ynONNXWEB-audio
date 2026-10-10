/** The measurements automatic settings choose from, in a browser. */

import type { Measurements } from "./resolver.js";

export const WEB_MEASUREMENTS: Measurements = {
  "small": {
    "failed": {
      "webgpu": [
        "mix_decoder-int4_encoder-int4"
      ]
    },
    "rtf": {
      "wasm": {
        "mix_decoder-float32_encoder-int4": 2.98,
        "mix_decoder-float32_encoder-int8": 2.72,
        "mix_decoder-int4_encoder-int4": 1.72,
        "mix_decoder-int4_encoder-int8": 1.73,
        "mix_decoder-int8_encoder-int8": 1.77
      },
      "webgpu": {
        "mix_decoder-float32_encoder-float16": 4.61
      }
    },
    "size_mb": {
      "mix_decoder-float32_encoder-float16": 711.8,
      "mix_decoder-float32_encoder-int4": 596.1,
      "mix_decoder-float32_encoder-int8": 635.9,
      "mix_decoder-int4_encoder-int4": 167.3,
      "mix_decoder-int4_encoder-int8": 207.1,
      "mix_decoder-int8_encoder-int8": 260.2
    },
    "wer": {
      "wasm": {
        "en": {
          "mix_decoder-float32_encoder-int4": 2.29,
          "mix_decoder-float32_encoder-int8": 2.39,
          "mix_decoder-int4_encoder-int4": 3.02,
          "mix_decoder-int4_encoder-int8": 2.19,
          "mix_decoder-int8_encoder-int8": 2.39
        }
      },
      "webgpu": {
        "en": {
          "mix_decoder-float32_encoder-float16": 1.98
        }
      }
    }
  }
};
