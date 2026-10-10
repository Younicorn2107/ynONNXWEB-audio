import { defineConfig } from "tsup";

// Two profiles, and the split is not cosmetic.
//
//  - `index` / `core` — consumed through a bundler or through the core's
//    `auto` entry, so onnxruntime and the core stay external and the host
//    resolves them once.
//
//  - `worker` — inlined AND `noExternal`, exactly as the core's own worker
//    profile is. A worker script served cross-origin cannot be constructed
//    with `new Worker(url)`, so the core fetches the text and builds the
//    worker from a Blob URL. Inside a blob worker `import.meta.url` IS the
//    blob URL, so a relative import resolves to nothing and a dynamic
//    `import("onnxruntime-web")` resolves to nothing either. One
//    self-contained file is the only shape that survives that trip.
//
//    This package built the worker under the first profile until now:
//    `external` with no `noExternal`, no `splitting: false`, no
//    `platform: "browser"`. The output opened with relative imports of
//    shared chunks and still contained `await import('onnxruntime-web')` —
//    which is to say the Blob path the core is designed around could never
//    have worked from here. The core forbids this for its own bundle with a
//    test; this package had no build test at all, which is why it went
//    unnoticed. `tests/unit/build.test.ts` now asserts it here too.
//
// No profile sets `clean` — tsup runs them in parallel and a clean races the
// others, which cost a .d.ts in the core package until a test caught it.
export default defineConfig([
  {
    entry: {
      index: "src/index.ts",
      core: "src/core.ts",
    },
    format: ["esm"],
    dts: true,
    treeshake: true,
    external: ["onnxruntime-web", "@youni/onnxweb-core"],
  },
  {
    // A Worker is constructed from a URL, not imported: it needs a file of
    // its own to point at, and it must not be tree-shaken into the main
    // bundle.
    entry: { worker: "src/worker/entry.ts" },
    format: ["esm"],
    // `platform: "browser"` matters: the default injects a `createRequire`
    // shim that a worker has no way to satisfy.
    platform: "browser",
    dts: true,
    treeshake: true,
    // No shared chunks: a chunk is a second file, and the blob has no origin
    // to fetch it from.
    splitting: false,
    // Everything in, onnxruntime included. That is what makes the file
    // survive being read as text and rebuilt from a Blob.
    noExternal: [/.*/],
  },
]);
