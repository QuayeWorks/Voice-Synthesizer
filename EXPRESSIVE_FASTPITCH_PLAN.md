# Expressive FastPitch feasibility and modification plan

## Current FastPitch implementation (observations)
- **Architecture**: Encoder/decoder FFTransformers with duration, pitch, and optional energy predictors; speaker embedding is optional but inactive for single speaker.【F:third_party/fastpitch/fastpitch/model.py†L109-L210】 Length regulation uses MAS-based hard attention converted to durations, then repeats encoder outputs to the decoder.【F:third_party/fastpitch/fastpitch/model.py†L242-L385】
- **Alignment**: Uses convolutional attention with Gaussian similarity and MAS binarization to derive durations from ground-truth mels; training returns attn_soft/logprob for losses.【F:third_party/fastpitch/fastpitch/model.py†L259-L339】【F:third_party/fastpitch/fastpitch/attention.py†L82-L220】
- **Inference path**: Predicts durations, pitch, (optional) energy; applies pitch/energy embeddings to encoder outputs; decoder projects to mels. Pitch transform hook exists for global shift but no fine-grained style control.【F:third_party/fastpitch/fastpitch/model.py†L387-L445】

## Viability assessment
FastPitch can remain the base model for expressive single-speaker narration with targeted additions. Its non-autoregressive decoder keeps inference fast, and the alignment/duration pipeline can support style conditioning. However, FastPitch lacks built-in style tokens, reference encoders, or stochastic duration/pitch sampling. With ~20 hours and 2×1080 Ti (11 GB each), the encoder/decoder FFT blocks are the main memory users; added style modules must be lightweight to preserve speed.

Alternative architectures (VITS/StyleTTS2/diffusion TTS) offer stronger prosody by default, but they incur slower inference or larger VRAM (flow + GAN vocoders) and more training instability. Given the constraint to keep FastPitch speed and hardware, extending FastPitch is viable; switching is only necessary if you require implicit emotional transfer without tags or reference audio.

## Proposed modifications (patch-level guidance)
### A) Style tokens/tags in text
- **Code changes**:
  - **Embedding table**: Extend `FastPitch.encoder` embedding (`FFTransformer(embed_input=True, n_embed=n_symbols, padding_idx=padding_idx)`) to include special style tokens; update symbol set and `padding_idx` in dataset/tokenizer config so tags map to new IDs.【F:third_party/fastpitch/fastpitch/model.py†L132-L210】
  - **Style conditioning**: Add a small learned style embedding lookup (e.g., `nn.Embedding(n_styles, symbols_embedding_dim)`) and sum it with `text_emb` before attention and encoder outputs. Insert in `FastPitch.forward` and `infer` right after `text_emb = self.encoder.word_emb(inputs)`; broadcast style ID per sequence. Shapes: style ID `[B] → style_emb [B, symbols_embedding_dim] → unsqueeze to [B,1,dim]` and add to `text_emb` `[B,T,dim]` and `enc_out` `[B,T,dim]` before predictors.
- **Inference selection**: Prefix text with `<shout>`, `<whisper>`, `<inner>`, etc.; tokenizer maps tags to IDs. Pace/pitch controls remain usable.
- **Training**: Tag each transcript line with style labels; MAS alignment unchanged because tokens are treated as normal symbols. Consider class-balanced sampling so rare styles appear each epoch.

### B) Reference prosody encoder (style embedding from audio)
- **Code changes**:
  - Add a lightweight reference encoder (Conv + GRU) module (e.g., in `fastpitch/model.py`) producing `[B, style_dim]` from reference mel `[B, n_mel, T_ref]`.
  - Introduce a style projection to match `symbols_embedding_dim` and add to `enc_out`/`text_emb` similar to style tokens. When reference is present, skip/additive combine with tag embedding.
  - Add arguments to `FastPitch.__init__` to enable/size the reference encoder and toggle during inference.
  - Update `infer` signature to accept `ref_mel` (or path) → encode → style conditioning.
- **Shapes**: ref mel `[B, n_mel, T] → conv stack → GRU → pooled [B, style_dim] → linear to `[B, symbols_embedding_dim]` → unsqueeze `[B,1,dim]` for addition.
- **Inference selection**: Provide a short reference wav/mel matching desired emotion; if both ref and tag provided, concatenate or sum embeddings; otherwise default neutral.
- **Training**: For each sample, feed ground-truth mel as reference (teacher-forced) or augment by pairing neutral text with expressive ref to teach disentanglement. MAS alignment unchanged (uses `mel_tgt`).

### C) Predictor upgrades (optional)
- **Duration distribution sampling**: Replace deterministic `dur_pred = clamp(exp(log_dur_pred)-1)` with sampling from a log-normal: draw `eps ~ N(0,1)` and set `dur_sample = clamp(exp(log_dur_pred + sigma*eps)-1)`. Add `sigma` hyperparam; keep argmax path for training stability. Implement in `infer` with optional flag to keep training losses intact.【F:third_party/fastpitch/fastpitch/model.py†L346-L405】
- **Pitch contour smoothing**: Add an L2 penalty on frame-level pitch deltas before averaging (`average_pitch`) to discourage jagged contours. Loss can be added in `loss_function.py` using `pitch_dense` and `dur_tgt` output from forward pass.【F:third_party/fastpitch/fastpitch/model.py†L350-L360】【F:third_party/fastpitch/fastpitch/loss_function.py†L36-L80】
- **Energy predictor**: Already present and optional; keep enabled for stronger loudness control.【F:third_party/fastpitch/fastpitch/model.py†L189-L205】

### D) Training sampling to keep extremes
- Upweight rare styles using per-style sampling probabilities in the dataset loader (outside the model). Mix neutral + expressive mini-batches; optionally freeze duration predictor for first N steps so MAS alignment stabilizes with new tokens.
- Use multi-style prompts per batch to prevent averaging; include short shouted/whisper clips as references for B).

## Speed and resource impact on 2×1080 Ti
- FastPitch remains fast; encoder/decoder FFT blocks dominate memory. Added embeddings and small reference encoder add negligible overhead compared to existing predictors and attention.
- Bottlenecks: self-attention activations in FFTransformer at longer sequences; MAS attention creation (`attn_soft`) scales with mel length and may increase memory when batching long, expressive lines.【F:third_party/fastpitch/fastpitch/model.py†L259-L385】 Use gradient checkpointing or shorter chunks if needed.
- Reference encoder adds a small convolution/GRU; keep style_dim ≤256 to fit VRAM. Disable energy predictor if memory is tight.

## Recommended paths
- **Minimal viable expressive FastPitch (fastest)**: Implement A) style tokens only. Add style embedding to encoder outputs; annotate transcripts with tags; use sampler to balance styles. Keeps current pipeline and maximal speed.
- **Best quality within FastPitch**: Implement A + B + pitch-smoothing loss from C. Use tag + reference conditioning, enable energy predictor, optional stochastic duration sampling at inference for variation. Still non-autoregressive and fast.
- **When to switch models**: If you need high-fidelity, reference-driven emotional transfer without explicit tags or want natural ad-libs/comedic pauses learned implicitly, FastPitch’s deterministic duration/pitch predictors may plateau. Architectures like VITS/StyleTTS2 provide variational or flow-based prosody modeling and waveform-level objectives but will be slower and heavier; they also need more VRAM for training and inference than dual 1080 Ti can comfortably provide.
