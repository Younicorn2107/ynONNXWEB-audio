# onnxweb-audio

Speech and sound in the browser: Whisper transcription, diarization, alignment, sound events.

- **Transcription**: Whisper (tiny to large-v3-turbo), long audio in sections, voice activity detection, word timestamps, streaming from the microphone, resumable runs.
- **Speakers**: pyannote segmentation and WeSpeaker embeddings, clustered into speakers.
- **Analysis**: emotion and AudioSet sound events.
- **Exports**: SRT, VTT, TXT, JSON, CSV, XML.

```ts
import * as ort from "onnxruntime-web";
import { AudioTranscriber, decodeAudio } from "@youni/onnxweb-audio";

const transcriber = await AudioTranscriber.load({ ort, Tensor: ort.Tensor, model: { name: "base", computeType: "int8" } });
const { audio } = await decodeAudio(file);
const result = await transcriber.transcribe(audio, { language: "fr" });
console.log(result.text);
```

Weights: Whisper (MIT), pyannote segmentation (MIT), WeSpeaker (CC-BY-4.0, attribution), AST (BSD-3-Clause).

## Install

Builds against `@youni/onnxweb-core`:

```sh
npm install
npm run build   # dist/
npm test        # unit tests (Node, onnxruntime-web on WebAssembly)
```

## Licence

Apache-2.0 for the code. Model weights keep their own licence (listed above where it matters).
