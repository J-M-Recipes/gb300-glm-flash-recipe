#!/usr/bin/env python3
"""router4.py — (1) Monolithic kernel routing at M=1 vs M=512 on the same logits (does tie-break depend on batch?);
(2) routing WEIGHTS: NoAuxTc(1,1) vs fused_topk_deepseek(0,0) vs vLLM grouped_topk, bit-exact after bf16 pack?
(3) full Monolithic output on real quantized inputs at M=1 vs the routed kernel fed NoAuxTc ids/weights (tie tokens)."""
import torch, json, glob
import flashinfer
from flashinfer.fused_moe import RoutingMethodType, fused_topk_deepseek
from flashinfer.fused_moe.fused_routing_dsv3 import get_dsv3_fused_routing_module
from vllm.model_executor.layers.fused_moe.router.grouped_topk_router import grouped_topk
from vllm.model_executor.layers.fused_moe.utils import trtllm_moe_pack_topk_ids_weights as pack
dev = "cuda"; E, K, HID, INTER = 256, 8, 6144, 2048
R = torch.load("/w/capture/L3_logit_ring.pt"); n = min(R["n"], R["ring"]); logits = R["logits"][:n].to(dev)
W = torch.load("/w/capture/L3_weights.pt", map_location=dev); bias = W["bias"].float().to(dev); act = W["activation_int"]
_d = get_dsv3_fused_routing_module()
def mono_route(lg, ng=1, tg=1):
    M = lg.shape[0]; rr = torch.full((M, K), -1, dtype=torch.int16, device=dev)
    x = torch.zeros(M, HID // 2, dtype=torch.uint8, device=dev); xs = torch.zeros(M, HID // 16, dtype=torch.float8_e4m3fn, device=dev)
    flashinfer.fused_moe.trtllm_fp4_block_scale_moe(routing_logits=lg.contiguous(), routing_bias=bias, hidden_states=x, hidden_states_scale=xs,
        gemm1_weights=W["w13"], gemm1_weights_scale=W["w13s"].view(torch.float8_e4m3fn), gemm1_bias=None, gemm1_alpha=None, gemm1_beta=None, gemm1_clamp_limit=None,
        gemm2_weights=W["w2"], gemm2_weights_scale=W["w2s"].view(torch.float8_e4m3fn), gemm2_bias=None,
        output1_scale_scalar=W["g1c"], output1_scale_gate_scalar=W["g1a"], output2_scale_scalar=W["g2a"],
        num_experts=E, top_k=K, n_group=ng, topk_group=tg, intermediate_size=INTER, local_expert_offset=0, local_num_experts=E,
        routed_scaling_factor=1.0, routing_method_type=int(RoutingMethodType.DeepSeekV3), do_finalize=True, activation_type=act,
        per_token_scale=None, tune_max_num_tokens=8, routing_replay_out=rr)
    return torch.sort(rr.to(torch.int32), 1)[0]
def noaux(lg, ng=1, tg=1):
    M = lg.shape[0]; tv = torch.empty(M, K, dtype=torch.float32, device=dev); ti = torch.empty(M, K, dtype=torch.int32, device=dev)
    _d.NoAuxTc(lg.contiguous(), bias, ng, tg, K, 1.0, tv, ti, False, None); return ti, tv
def ftd(lg):   # only (1,1) is a valid no-group config for 256 experts; second call = repeatability check
    return noaux(lg, 1, 1)
def vllm(lg):
    x = torch.zeros(lg.shape[0], HID, dtype=torch.bfloat16, device=dev)
    w, i = grouped_topk(x, lg, K, True, 1, 1, "sigmoid", 1.0, bias); return i.to(torch.int32), w.float()
out = {"tokens": n}
sc = torch.sigmoid(logits) + bias[None, :]; top9 = torch.topk(sc, 9, 1).values; gap = top9[:, 7] - top9[:, 8]
ties = (gap == 0).nonzero().flatten().tolist(); out["tie_tokens"] = len(ties)
# (1) batch dependence of kernel routing
b512 = torch.cat([mono_route(logits[s:s+512]) for s in range(0, n, 512)])
sample = sorted(set(ties + list(range(64))))
m1 = torch.cat([mono_route(logits[i:i+1]) for i in sample])
out["kernel_M1_vs_M512_mismatch"] = int((m1 != b512[sample]).any(1).sum()); out["sampled"] = len(sample)
m1_ties = torch.cat([mono_route(logits[i:i+1]) for i in ties]); na_ties = torch.sort(noaux(logits[ties])[0], 1)[0]
out["kernel_M1_vs_noaux_on_ties"] = int((m1_ties != na_ties).any(1).sum())
# (2) weights
ia, wa = noaux(logits); ib, wb = ftd(logits); ic, wc = vllm(logits)
def sortw(i, w):
    o = torch.argsort(i, 1); return torch.gather(i, 1, o), torch.gather(w, 1, o)
ia, wa = sortw(ia, wa); ib, wb = sortw(ib, wb); ic, wc = sortw(ic, wc)
same_ab = (ia == ib).all(1); same_ac = (ia == ic).all(1)
out["ids_noaux_vs_ftd_mismatch"] = int((~same_ab).sum()); out["ids_noaux_vs_vllm_mismatch"] = int((~same_ac).sum())
out["w_fp32_noaux_vs_ftd_maxabs_sameids"] = float((wa[same_ab] - wb[same_ab]).abs().max())
out["w_fp32_noaux_vs_vllm_maxabs_sameids"] = float((wa[same_ac] - wc[same_ac]).abs().max())
out["w_bf16_noaux_vs_vllm_neq_sameids"] = int((wa[same_ac].bfloat16() != wc[same_ac].bfloat16()).sum())
out["w_bf16_noaux_vs_ftd_neq_sameids"] = int((wa[same_ab].bfloat16() != wb[same_ab].bfloat16()).sum())
# (3) real decode inputs: Monolithic (its own routing) vs routed kernel with NoAuxTc ids/weights, M=1 each
files = sorted(f for f in glob.glob("/w/capture/L3_*.pt") if "weights" not in f and "logit" not in f)
n3 = 0; neq3 = 0; maxabs3 = 0.0
for f in files:
    c = torch.load(f, map_location=dev)
    lg = c["router_logits"].float() if "router_logits" in c else None
    if lg is None: continue
    for t in range(lg.shape[0]):
        x = c["hidden_states"][t:t+1]; xs = c["a1q_scale"][t:t+1] if c["a1q_scale"].dim() > 1 else c["a1q_scale"]
        M = 1; rr = torch.full((M, K), -1, dtype=torch.int16, device=dev)
        ref = flashinfer.fused_moe.trtllm_fp4_block_scale_moe(routing_logits=lg[t:t+1].contiguous(), routing_bias=bias, hidden_states=x, hidden_states_scale=xs.view(torch.float8_e4m3fn).reshape(1, -1),
            gemm1_weights=W["w13"], gemm1_weights_scale=W["w13s"].view(torch.float8_e4m3fn), gemm1_bias=None, gemm1_alpha=None, gemm1_beta=None, gemm1_clamp_limit=None,
            gemm2_weights=W["w2"], gemm2_weights_scale=W["w2s"].view(torch.float8_e4m3fn), gemm2_bias=None,
            output1_scale_scalar=W["g1c"], output1_scale_gate_scalar=W["g1a"], output2_scale_scalar=W["g2a"],
            num_experts=E, top_k=K, n_group=1, topk_group=1, intermediate_size=INTER, local_expert_offset=0, local_num_experts=E,
            routed_scaling_factor=1.0, routing_method_type=int(RoutingMethodType.DeepSeekV3), do_finalize=True, activation_type=act,
            per_token_scale=None, tune_max_num_tokens=8, routing_replay_out=rr)
        ti, tv = noaux(lg[t:t+1]); packed = pack(ti, tv)
        o = torch.empty(1, HID, dtype=torch.bfloat16, device=dev)
        flashinfer.fused_moe.trtllm_fp4_block_scale_routed_moe(topk_ids=packed, routing_bias=None, hidden_states=x, hidden_states_scale=xs.view(torch.float8_e4m3fn).reshape(1, -1),
            gemm1_weights=W["w13"], gemm1_weights_scale=W["w13s"].view(torch.float8_e4m3fn), gemm1_bias=None, gemm1_alpha=None, gemm1_beta=None, gemm1_clamp_limit=None,
            gemm2_weights=W["w2"], gemm2_weights_scale=W["w2s"].view(torch.float8_e4m3fn), gemm2_bias=None,
            output1_scale_scalar=W["g1c"], output1_scale_gate_scalar=W["g1a"], output2_scale_scalar=W["g2a"],
            num_experts=E, top_k=K, n_group=0, topk_group=0, intermediate_size=INTER, local_expert_offset=0, local_num_experts=E,
            routed_scaling_factor=None, routing_method_type=1, do_finalize=True, activation_type=act, per_token_scale=None, output=o, tune_max_num_tokens=8)
        ref = ref[0] if isinstance(ref, (list, tuple)) else ref
        d = (ref.float() - o.float()).abs().max().item(); n3 += 1; neq3 += int(d > 0); maxabs3 = max(maxabs3, d)
out["real_M1_mono_vs_routed_noaux"] = {"n": n3, "neq": neq3, "maxabs": maxabs3}
print("ROUTER4 " + json.dumps(out))
