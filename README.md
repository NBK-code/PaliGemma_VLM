# PaliGemma — From-Scratch PyTorch Implementation

This repository contains a **from-scratch PyTorch implementation of the PaliGemma vision–language model**, closely following the official Google Research and Hugging Face reference implementations.

The primary goal of this project is **implementation fidelity and transparency**. All components are written explicitly to make the model architecture, data flow, and inference mechanics easy to inspect and reason about.

This codebase is intended for **learning, research, and controlled inference experiments**, rather than as a production-ready system.

---

## Key Features

- Faithful implementation of the **SigLIP vision encoder**
- **Gemma causal language model** with:
  - Gemma-style RMSNorm
  - Grouped Query Attention (GQA)
  - Rotary Positional Embeddings (RoPE)
- Explicit **KV cache** for efficient autoregressive decoding
- Multimodal fusion via **image-token replacement**
- Inference behavior aligned with the official PaliGemma reference

---

## Image Processing

Images are processed using the following steps:

- Resize to `(image_size, image_size)`
- Rescale pixel values to the range `[0, 1]`
- Normalize using:
  - Mean: `[0.5, 0.5, 0.5]`
  - Standard deviation: `[0.5, 0.5, 0.5]`
- Convert to channel-first format `(C, H, W)`

The processed images are encoded using the SigLIP vision transformer and then projected into the text model’s hidden space.

---

## Text Processing

Text inputs are formatted to match the PaliGemma training and inference setup:

- A fixed number of `<image>` tokens are **prepended** to the prompt
- The `<bos>` token is inserted manually
- A newline character (`\n`) is appended explicitly
- Automatic BOS/EOS insertion by the tokenizer is disabled

Example input format:

```<image><image>...<image><bos>Your prompt here\n```
---

## Multimodal Fusion

- `<image>` token positions in the token sequence are replaced with projected image patch embeddings
- Text token embeddings are preserved
- Padding tokens (if any) are masked out
- Image features are scaled by `1 / sqrt(hidden_size)` before fusion

This behavior mirrors the reference inference logic used in the official implementation.

---

## Inference Assumptions and Limitations

This implementation intentionally follows the **reference inference path** and therefore assumes:

- No padding tokens (all attention mask values are `1`)
- One image and one text prompt per forward pass
- Inference-only usage (no training loss implemented)

These constraints simplify the code while remaining faithful to the official inference setup.

---

## References

- **PaliGemma: A Family of Vision-Language Models**  
  Google Research, 2024  
  https://arxiv.org/abs/2407.07726

- **SigLIP: Sigmoid Loss for Language Image Pre-training**  
  Zhai et al.  
  https://arxiv.org/abs/2303.15343

- **Gemma: Open Models Based on Gemini Research**  
  Google DeepMind  
  https://arxiv.org/abs/2403.08295

- **Hugging Face PaliGemma Model Page**  
  https://huggingface.co/google/paligemma-3b-pt-224

- **Hugging Face Transformers – PaliGemma Implementation**  
  https://github.com/huggingface/transformers/tree/main/src/transformers/models/paligemma

- **Umair Jamil's Tutorial**  
  https://www.youtube.com/watch?v=vAmKB7iPkWw

---
