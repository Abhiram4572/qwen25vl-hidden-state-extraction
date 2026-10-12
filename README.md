# Qwen2.5-VL-7B Hidden State Extraction

Extract internal hidden-state representations from `Qwen/Qwen2.5-VL-7B-Instruct`:

- The residual-stream hidden state at layer `l`, position `t`.
- The per-head attention output at layer `l`, position `t` (captured right before `o_proj` merges heads).

## Setup

```bash
pip install torch transformers accelerate pillow
```

## Usage

```bash
python extract_hidden_states.py --image path/to/img.jpg --prompt "What is in the image?" --layer 10 --position -1
```

Text-only (no image):

```bash
python extract_hidden_states.py --prompt "Describe a sunset." --layer 5 --position -1
```

### Flags

| Flag | Default | Description |
|---|---|---|
| `--model` | `Qwen/Qwen2.5-VL-7B-Instruct` | HF model id |
| `--image` | `None` | Path to an image (omit for text-only) |
| `--prompt` | `"Describe the image."` | Text prompt |
| `--layer` | `10` | Layer `l` to extract (0-indexed) |
| `--position` | `-1` | Token position `t` (negative indexes from the end) |
| `--device` | `cuda` if available, else `cpu` | Device to run on |

Run `python extract_hidden_states.py --help` for the full list.

## Output

On run, prints:
- Model overview: number of layers, hidden size, attention heads, KV heads (GQA), head dim, vocab size.
- Sequence length and resolved position index.
- The residual hidden state vector shape and first few values at `(l, t)`.
- The per-head attention output shape `[num_heads, head_dim]` and first few values per head at `(l, t)`.

## Note

Qwen2.5-VL's image patch 0 (top-left) acts as a positional attention sink, not a semantically meaningful token — exclude it from any top-k image-patch analysis built on top of this extraction.
