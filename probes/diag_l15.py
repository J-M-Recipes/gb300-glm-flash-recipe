#!/usr/bin/env python3
"""diag_l15.py — on the captured decode step (prompt 12, step 3), for layer 15:
   1. run the Monolithic kernel offline on the captured MLP input with checkpoint weights (reference)
   2. run the routed kernel with weights gathered by expert id from the checkpoint (= what a correct slot cache does)
   compare each against the live V1 output and live sc9 output. Whichever matches tells us if the live sc9 is
   serving wrong slot contents (stale/overwritten) or if the routed kernel itself differs at this input."""
import sys, torch, json
sys.path.insert(0, "/w"); L = 15
A = torch.load("/w/trace/d12-v1/layer_trace.pt", map_location="cuda"); B = torch.load("/w/trace/d12-sc9/layer_trace.pt", map_location="cuda")
fa, fb = A["full"], B["full"]
xin_a, xin_b = fa[f"L{L:02d}.mlp.gate.in"], fb[f"L{L:02d}.mlp.gate.in"]
print("MLP input identical:", torch.equal(xin_a, xin_b), tuple(xin_a.shape), xin_a.dtype)
ga, gb = fa[f"L{L:02d}.mlp.gate"], fb[f"L{L:02d}.mlp.gate"]
print("gate logits identical:", torch.equal(ga, gb))
ea, eb = fa[f"L{L:02d}.mlp.experts"], fb[f"L{L:02d}.mlp.experts"]
d = (ea.float() - eb.float()); print("live experts out: neq", int((d != 0).sum()), "maxabs", float(d.abs().max()))
# expert ids via TRT NoAuxTc (same as both builds' effective routing)
import flashinfer
from ablate_layer import build_layer_weights, run_mono, run_routed   # reuse last night's helpers
w = build_layer_weights(L)
ref = run_mono(w, xin_a)                                # Monolithic on this exact input
routed_full = run_routed(w, xin_a, gather_by_id=True)  # routed kernel, correct per-id weights (ideal cache)
for name, y in (("mono_offline", ref), ("routed_offline_correct_weights", routed_full)):
    for lbl, live in (("live_v1", ea), ("live_sc9", eb)):
        dd = (y.float() - live.float()); print(f"{name:32s} vs {lbl:9s}: neq={int((dd!=0).sum()):5d} maxabs={float(dd.abs().max()):.3e}")
ids = flashinfer.fused_moe.fused_topk_deepseek if False else None
print("topk ids (NoAuxTc):", w["route"](ga.float())[:8].tolist() if "route" in w else "n/a")
