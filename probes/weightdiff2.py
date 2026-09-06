#!/usr/bin/env python3
"""weightdiff2.py — run vLLM's REAL modelopt NVFP4 MoE weight pipeline on layer-3 raw weights, for BOTH kernel
classes, and diff every kernel-format tensor against (a) each other and (b) what the hook produced (captured from
cap-modular). If the two classes' process_weights_after_loading differ, we found it. If both match each other but
not the capture, the hook's pipeline differs."""
import os, torch
os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")
from vllm.model_executor.layers.fused_moe.oracle.nvfp4 import convert_to_nvfp4_moe_kernel_format, NvFp4MoeBackend, make_nvfp4_moe_quant_config
from vllm.model_executor.layers.fused_moe.experts.trtllm_nvfp4_moe import TrtLlmNvFp4ExpertsModular, TrtLlmNvFp4ExpertsMonolithic

dev = "cuda"
raw = torch.load("/w/capture/L3_raw.pt", map_location=dev)
cap = torch.load("/w/capture/L3_weights.pt", map_location=dev)

from vllm.model_executor.layers.fused_moe.config import FusedMoEConfig, FusedMoEParallelConfig, RoutingMethodType, MoEActivation
_pc = FusedMoEParallelConfig(tp_size=1, pcp_size=1, dp_size=1, ep_size=1, tp_rank=0, pcp_rank=0, dp_rank=0, ep_rank=0, sp_size=1, use_ep=False, all2all_backend="naive", enable_eplb=False)
_mc = FusedMoEConfig(num_experts=256, experts_per_token=8, hidden_dim=6144, intermediate_size=2048, num_local_experts=256, num_logical_experts=256,
                     activation=MoEActivation.SILU, device=torch.device("cuda"), routing_method=RoutingMethodType.DeepSeekV3, moe_parallel_config=_pc,
                     in_dtype=torch.bfloat16, router_logits_dtype=torch.float32)
class L(torch.nn.Module):  # minimal RoutedExperts stand-in with the attributes convert_* reads
    def __init__(self):
        super().__init__(); self.activation = MoEActivation.SILU; self.moe_config = _mc

def run(experts_cls):
    layer = L()
    layer.w13_weight = torch.nn.Parameter(raw["w13"].clone(), requires_grad=False)
    layer.w13_weight_scale = torch.nn.Parameter(raw["w13s"].clone(), requires_grad=False)
    layer.w13_weight_scale_2 = torch.nn.Parameter(raw["w13s2"].clone(), requires_grad=False)
    layer.w13_input_scale = torch.nn.Parameter(raw["a13"].clone(), requires_grad=False)
    layer.w2_weight = torch.nn.Parameter(raw["w2"].clone(), requires_grad=False)
    layer.w2_weight_scale = torch.nn.Parameter(raw["w2s"].clone(), requires_grad=False)
    layer.w2_weight_scale_2 = torch.nn.Parameter(raw["w2s2"].clone(), requires_grad=False)
    layer.w2_input_scale = torch.nn.Parameter(raw["a2"].clone(), requires_grad=False)
    w13s2 = layer.w13_weight_scale_2[:, 0].contiguous()
    out = convert_to_nvfp4_moe_kernel_format(nvfp4_backend=NvFp4MoeBackend.FLASHINFER_TRTLLM, layer=layer,
        w13=layer.w13_weight, w13_scale=layer.w13_weight_scale, w13_scale_2=w13s2, a13_scale=layer.w13_input_scale,
        w2=layer.w2_weight, w2_scale=layer.w2_weight_scale, w2_scale_2=layer.w2_weight_scale_2, a2_scale=layer.w2_input_scale,
        is_act_and_mul=True)
    w13, w13_scale, w13_scale_2, a13_scale, w2, w2_scale, w2_scale_2, a2_scale = out
    for n, t in zip(["w13_weight","w13_weight_scale","w13_weight_scale_2","w13_input_scale","w2_weight","w2_weight_scale","w2_weight_scale_2","w2_input_scale"], out):
        setattr(layer, n, torch.nn.Parameter(t if isinstance(t, torch.Tensor) else t, requires_grad=False))
    qc = make_nvfp4_moe_quant_config(backend=NvFp4MoeBackend.FLASHINFER_TRTLLM, w13_scale=layer.w13_weight_scale, w2_scale=layer.w2_weight_scale,
        w13_scale_2=layer.w13_weight_scale_2, w2_scale_2=layer.w2_weight_scale_2, a13_scale=layer.w13_input_scale, a2_scale=layer.w2_input_scale,
        swiglu_limit=None, swiglu_alpha=None, swiglu_beta=None, layer=layer)
    mc = _mc
    ex = experts_cls(moe_config=mc, quant_config=qc)
    ex.process_weights_after_loading(layer)
    return dict(w13=layer.w13_weight.data, w2=layer.w2_weight.data, w13s=layer.w13_weight_scale.data, w2s=layer.w2_weight_scale.data,
                g1c=layer.g1_scale_c.data, g1a=qc.g1_alphas, g2a=qc.g2_alphas, a1g=qc.a1_gscale, a2g=qc.a2_gscale)

def diff(a, b, label):
    for k in ("w13","w2","w13s","w2s","g1c","g1a","g2a"):
        if k not in a or k not in b or a[k] is None or b[k] is None: print(f"  {label} {k}: missing"); continue
        x, y = a[k], b[k]
        if x.shape != y.shape: print(f"  {label} {k}: SHAPE {tuple(x.shape)} vs {tuple(y.shape)}"); continue
        xb = x.view(torch.uint8) if x.element_size() == 1 else x.view(torch.int32) if x.element_size() == 4 else x
        yb = y.view(torch.uint8) if y.element_size() == 1 else y.view(torch.int32) if y.element_size() == 4 else y
        neq = int((xb != yb).sum())
        extra = ""
        if k in ("g1c","g1a","g2a") and neq:
            extra = f" maxrel={float(((x.float()-y.float()).abs()/(y.float().abs()+1e-30)).max()):.3e} x[:3]={x[:3].tolist()} y[:3]={y[:3].tolist()}"
        print(f"  {label} {k}: {'BIT-EXACT' if neq == 0 else f'{neq} elems differ'}{extra}")

mono = run(TrtLlmNvFp4ExpertsMonolithic); mod = run(TrtLlmNvFp4ExpertsModular)
print("== Monolithic vs Modular (stock pipeline, same raw weights)"); diff(mono, mod, "mono~mod")
print("== stock Modular vs hook capture"); diff(mod, cap, "mod~cap")
print("== stock Monolithic vs hook capture"); diff(mono, cap, "mono~cap")
