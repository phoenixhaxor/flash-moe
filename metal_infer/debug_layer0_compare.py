#!/usr/bin/env python3
"""Compare hidden state after layer 0: Python (MLX) vs C (infer.m)"""
import numpy as np
import struct
import glob
from mlx_lm.utils import load

model_path = "/Users/andre/.cache/huggingface/hub/models--mlx-community--Qwen3.6-35B-A3B-8bit/snapshots/q36"

# Load the full MLX model
print("Loading model...")
model, tokenizer = load(model_path)
print(f"Model loaded. Vocab size: {model.args.vocab_size}")

# Tokenize "Hello"
tokens = tokenizer.encode("Hello")
print(f"Token IDs: {tokens}")

# Get embedding for first token
token_id = tokens[0]
embed_weight = model.model.embed_tokens.weight
hidden = embed_weight[token_id]
print(f"Embedding shape: {hidden.shape}")
print(f"Embedding stats: mean={float(mx.mean(hidden)):.6f} std={float(mx.std(hidden)):.6f}")

# Forward through layer 0
layer0 = model.model.layers[0]
print(f"\nLayer 0 type: {type(layer0)}")
print(f"Layer 0 attributes: {[a for a in dir(layer0) if not a.startswith('_')][:20]}")
