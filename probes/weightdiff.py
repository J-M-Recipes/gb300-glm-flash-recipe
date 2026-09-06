#!/usr/bin/env python3
"""weightdiff.py — do the hook-processed layer-3 kernel-format weights (captured from cap-modular) equal what a STOCK
process_weights_after_loading produces from the checkpoint? Runs offline: loads layer 3's raw safetensors, builds a
RoutedExperts-equivalent via the same modelopt path vLLM uses, and compares every kernel-format tensor bit-for-bit."""
import os, json, glob, torch, safetensors.torch as st
from safetensors import safe_open

dev = "cuda"
cap = torch.load("/w/capture/L3_weights.pt", map_location=dev)
print("captured:", {k: (tuple(v.shape), str(v.dtype)) if hasattr(v, "shape") else v for k, v in cap.items()})

# --- locate layer-3 expert tensors in the checkpoint index
idx = json.load(open("/model/model.safetensors.index.json"))["weight_map"]
pref = "model.layers.3.mlp.experts."
keys = sorted(k for k in idx if k.startswith(pref))
files = sorted({idx[k] for k in keys})
print("layer3 expert keys:", len(keys), "files:", files[:3], "…")
# sample the key naming
print("sample keys:", keys[:4])
raw = {}
for f in files:
    with safe_open(f"/model/{f}", "pt", device="cpu") as fh:
        for k in fh.keys():
            if k.startswith(pref): raw[k] = fh.get_tensor(k)
print("raw tensors:", len(raw))
# per-expert naming: model.layers.3.mlp.experts.{e}.{gate_proj|up_proj|down_proj}.{weight|weight_scale|weight_scale_2|input_scale}
import re
E = 1 + max(int(re.match(pref + r"(\d+)\.", k).group(1)) for k in raw if re.match(pref + r"\d+\.", k))
print("E", E)
def stack(name):
    return torch.stack([raw[f"{pref}{e}.{name}"] for e in range(E)])
w1 = stack("gate_proj.weight"); w3 = stack("up_proj.weight"); w2 = stack("down_proj.weight")
w1s = stack("gate_proj.weight_scale"); w3s = stack("up_proj.weight_scale"); w2s = stack("down_proj.weight_scale")
w1s2 = stack("gate_proj.weight_scale_2"); w3s2 = stack("up_proj.weight_scale_2"); w2s2 = stack("down_proj.weight_scale_2")
a1 = stack("gate_proj.input_scale"); a3 = stack("up_proj.input_scale"); a2 = stack("down_proj.input_scale")
print("raw shapes: w1", tuple(w1.shape), w1.dtype, "| w1s", tuple(w1s.shape), w1s.dtype, "| w1s2", tuple(w1s2.shape), "| a1", tuple(a1.shape))
# vLLM's loader concatenates gate+up along dim 0 into w13 ([E, 2*I, H/2] u8), scales likewise; input_scale = max(a1,a3)
w13 = torch.cat([w1, w3], dim=1); w13s = torch.cat([w1s, w3s], dim=1)
w13s2 = torch.stack([w1s2.reshape(E), w3s2.reshape(E)], dim=1)   # [E,2] as vLLM stores w13_weight_scale_2
a13 = torch.maximum(a1.reshape(E), a3.reshape(E))
print("w13", tuple(w13.shape), "w13s", tuple(w13s.shape))
torch.save(dict(w13=w13, w13s=w13s, w13s2=w13s2, a13=a13, w2=w2, w2s=w2s, w2s2=w2s2.reshape(E), a2=a2.reshape(E)), "/w/capture/L3_raw.pt")
print("RAW_SAVED")
