---
title: Pocket TTS ONNX Web Demo
emoji: 🌖
colorFrom: yellow
colorTo: pink
sdk: static
app_file: index.html
pinned: false
license: cc-by-4.0
short_description: Real-time voice cloning entirely in your browser! (CPU)
models:
  - KevinAHM/pocket-tts-onnx
custom_headers:
  cross-origin-embedder-policy: require-corp
  cross-origin-opener-policy: same-origin
  cross-origin-resource-policy: cross-origin
---

# Pocket TTS Web Demo

Real-time neural text-to-speech with voice cloning, running entirely in your browser.

## Features

- **Voice Cloning**: Clone any voice from a short audio sample
- **Predefined Voices**: 3 bundled voices (Cosette, Jean, Fantine)
- **Streaming Audio**: Real-time audio generation with low latency
- **Pure Browser**: No server required, runs entirely in WebAssembly

## Model Files

The demo requires the following ONNX models in the `onnx/` directory:

| File | Size | Purpose |
|------|------|---------|
| `mimi_encoder.onnx` | ~70 MB | Voice audio → embeddings |
| `text_conditioner.onnx` | ~16 MB | Text tokens → embeddings |
| `flow_lm_main_int8.onnx` | ~73 MB | AR transformer (INT8) |
| `flow_lm_flow_int8.onnx` | ~10 MB | Flow matching network (INT8) |
| `mimi_decoder_int8.onnx` | ~22 MB | Latents → audio decoder (INT8) |

Additional files:
- `tokenizer.model` - SentencePiece tokenizer (~60 KB)
- `voices.bin` - Predefined voice embeddings (~1.5 MB)

## Browser Requirements

- Modern browser with WebAssembly support
- Chrome, Edge, Firefox, or Safari (latest versions)
- ~200 MB RAM for model loading

## Voice Cloning

1. Click "Upload Voice" or select "Custom (Upload)" from the dropdown
2. Upload an audio file (WAV, MP3, etc.) with clear speech
3. Best results with 3-10 seconds of clean audio
4. The voice will be encoded and used for all subsequent generations

## File Structure

```
pocket-tts-web/
├── index.html              # Main HTML page
├── onnx-streaming.js       # Main thread controller
├── inference-worker.js     # Web Worker for ONNX inference
├── PCMPlayerWorklet.js     # Audio playback worklet
├── EventEmitter.js         # Event utilities
├── sentencepiece.js        # SentencePiece tokenizer library
├── style.css               # Styles
├── tokenizer.model         # SentencePiece model
├── voices.bin              # Predefined voice embeddings
└── onnx/
    ├── mimi_encoder.onnx
    ├── text_conditioner.onnx
    ├── flow_lm_main_int8.onnx
    ├── flow_lm_flow_int8.onnx
    └── mimi_decoder_int8.onnx
```

## License

- **Models & Voice Embeddings**: CC BY 4.0 (inherited from [kyutai/pocket-tts](https://huggingface.co/kyutai/pocket-tts))
- **Code**: Apache 2.0
