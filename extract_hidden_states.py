"""
Extract internal hidden-state representations from Qwen2.5-VL-7B:
  - residual-stream hidden state at layer l, position t
  - per-head attention output at layer l, position t (before o_proj)

Run: python extract_hidden_states.py --image path/to/img.jpg --prompt "Describe the image."
"""

import argparse
from dataclasses import dataclass

import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor


@dataclass
class LayerCapture:
    layer_idx: int
    attn_module: torch.nn.Module
    per_head_output: torch.Tensor = None  # [batch, num_heads, seq, head_dim]


class PerHeadOutputHook:
    """Captures the attention output *before* it gets merged by o_proj,
    i.e. reshaped as [batch, seq, num_heads, head_dim]."""

    def __init__(self, num_heads: int, head_dim: int):
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.output = None

    def __call__(self, module, inputs, output):
        # `inputs[0]` to o_proj is the concatenated per-head context: [batch, seq, hidden]
        hidden_in = inputs[0]
        batch, seq, hidden = hidden_in.shape
        self.output = hidden_in.view(batch, seq, self.num_heads, self.head_dim)


def build_model(model_name: str, dtype=torch.bfloat16, device="cuda"):
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name, torch_dtype=dtype, device_map=device
    )
    processor = AutoProcessor.from_pretrained(model_name)
    model.eval()
    return model, processor


def print_model_overview(model):
    cfg = model.config.text_config if hasattr(model.config, "text_config") else model.config
    num_layers = cfg.num_hidden_layers
    num_heads = cfg.num_attention_heads
    num_kv_heads = getattr(cfg, "num_key_value_heads", num_heads)
    hidden_size = cfg.hidden_size
    head_dim = hidden_size // num_heads

    print("=" * 60)
    print("Qwen2.5-VL-7B overview")
    print("=" * 60)
    print(f"Num hidden layers:      {num_layers}")
    print(f"Hidden size:            {hidden_size}")
    print(f"Num attention heads:    {num_heads}")
    print(f"Num KV heads (GQA):     {num_kv_heads}")
    print(f"Head dim:               {head_dim}")
    print(f"Vocab size:             {cfg.vocab_size}")
    print("=" * 60)
    return num_layers, num_heads, head_dim


def register_per_head_hooks(model, num_heads, head_dim):
    """Attach a forward hook to each decoder layer's o_proj to grab
    per-head attention outputs before they're merged."""
    hooks = {}
    handles = []
    layers = model.model.language_model.layers if hasattr(model.model, "language_model") else model.model.layers
    for l, layer in enumerate(layers):
        hook = PerHeadOutputHook(num_heads, head_dim)
        handle = layer.self_attn.o_proj.register_forward_hook(hook)
        hooks[l] = hook
        handles.append(handle)
    return hooks, handles


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--image", default=None, help="Path or URL to an image (optional)")
    parser.add_argument("--prompt", default="Describe the image.")
    parser.add_argument("--layer", type=int, default=10, help="Layer l to extract")
    parser.add_argument("--position", type=int, default=-1, help="Token position t (negative = from end)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    model, processor = build_model(args.model, device=args.device)
    num_layers, num_heads, head_dim = print_model_overview(model)

    hooks, handles = register_per_head_hooks(model, num_heads, head_dim)

    messages = [{"role": "user", "content": []}]
    if args.image:
        messages[0]["content"].append({"type": "image", "image": args.image})
    messages[0]["content"].append({"type": "text", "text": args.prompt})

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    if args.image:
        from PIL import Image
        image = Image.open(args.image).convert("RGB") if not args.image.startswith("http") else None
        inputs = processor(text=[text], images=[image] if image else None, return_tensors="pt").to(args.device)
    else:
        inputs = processor(text=[text], return_tensors="pt").to(args.device)

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True, output_attentions=True)

    seq_len = inputs["input_ids"].shape[1]
    l, t = args.layer, args.position
    t_idx = t if t >= 0 else seq_len + t

    print(f"\nSequence length: {seq_len}")
    print(f"Extracting layer l={l}, position t={t_idx}")

    # (1) residual-stream hidden state at layer l, position t
    # hidden_states[0] = embeddings, hidden_states[i] = output of layer i-1
    residual_state = outputs.hidden_states[l + 1][0, t_idx]  # [hidden_size]
    print(f"\nResidual hidden state shape: {residual_state.shape}")
    print(f"Residual hidden state (first 8 dims): {residual_state[:8].float().cpu().numpy()}")

    # (2) per-head attention output at layer l, position t (pre-o_proj)
    per_head = hooks[l].output  # [batch, seq, num_heads, head_dim]
    if per_head is not None:
        head_outputs_t = per_head[0, t_idx]  # [num_heads, head_dim]
        print(f"\nPer-head attention output shape: {head_outputs_t.shape}")
        for h in range(min(num_heads, 4)):
            print(f"  head {h} (first 4 dims): {head_outputs_t[h, :4].float().cpu().numpy()}")
        print(f"  ... ({num_heads} heads total)")
    else:
        print("Per-head output hook did not fire (check layer indexing).")

    for handle in handles:
        handle.remove()


if __name__ == "__main__":
    main()
