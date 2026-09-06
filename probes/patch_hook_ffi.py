#!/usr/bin/env python3
"""patch_hook_ffi.py — add SLOT_CACHE_ROUTER=ffi to slot_cache_hook.py (box v6 base): route with Monolithic's own
routing kernel (Routing::Runner::run from the loaded fused_moe_trtllm_sm100.so, via ffi_route.py). Narrow diff:
one import, one workspace cache, one branch in the runner seam next to the existing 'trt' branch."""
import sys
p = sys.argv[1]; s = open(p).read()

old_hdr = '_ROUTER = os.environ.get("SLOT_CACHE_ROUTER", "vllm")          # vllm | trt\n'
new_hdr = '''_ROUTER = os.environ.get("SLOT_CACHE_ROUTER", "vllm")          # vllm | trt | ffi
if _ROUTER == "ffi":
    sys.path.insert(0, os.path.dirname(os.path.abspath(os.environ.get("SLOT_CACHE_HOOK", "/w/slot_cache_hook.py"))))
    import ffi_route as _ffi_route                                   # Monolithic routing pass via exported symbol (0/3041 vs Monolithic)
    _FFI_WS = {}
    def _ffi_ws(M, K, E, device):
        ws = _FFI_WS.get(M)
        if ws is None:
            ws = _FFI_WS[M] = _ffi_route.RouteWorkspace(M, K, E, device)
        return ws
'''
assert s.count(old_hdr) == 1, "header anchor"; s = s.replace(old_hdr, new_hdr)
if "import sys" not in s.split("\n_ROUTER")[0]:
    s = s.replace("import os", "import os, sys", 1)

old_branch = '            if _ROUTER == "trt" and not self.routed_experts.quant_method.is_monolithic:\n'
new_branch = '''            if _ROUTER == "ffi" and not self.routed_experts.quant_method.is_monolithic:
                # Route with the Monolithic kernel's OWN routing pass (same compiled tanhf/normalise/bf16 cast as V1):
                # ids + bf16 weights bit-identical to Monolithic on 3041/3041 real tokens; routed kernel then bit-exact.
                # Scale 1.0 in-kernel because the runner applies routed_scaling_factor (2.5) to the output, as for V1.
                M = router_logits.shape[0]; K = int(getattr(rt, "top_k", 8)); E = router_logits.shape[1]
                ws = _ffi_ws(M, K, E, router_logits.device)
                _packed, wts, rr = _ffi_route.mono_route(router_logits.contiguous(), bias.contiguous(), ws,
                                                          routed_scaling_factor=1.0, n_group=1, topk_group=1)
                tv = wts.float(); ti = rr.to(torch.int32)
                self._maybe_apply_shared_experts(shared_experts_input, module.SharedExpertsOrder.NO_OVERLAP)
                fused_out = self.routed_experts.forward_modular(x=hidden_states, topk_weights=tv, topk_ids=ti,
                                                                shared_experts=self._shared_experts, shared_experts_input=shared_experts_input)
                self._maybe_apply_shared_experts(shared_experts_input, module.SharedExpertsOrder.MULTI_STREAM_OVERLAPPED)
                return (self._shared_experts.output if self._shared_experts is not None else None), fused_out
            if _ROUTER == "trt" and not self.routed_experts.quant_method.is_monolithic:
'''
assert s.count(old_branch) == 1, "branch anchor"; s = s.replace(old_branch, new_branch)
old_gate = '        if _CAP_N > 0 or _ROUTER == "trt" or _LOGIT_RING > 0:\n'
assert s.count(old_gate) == 1, "gate anchor"; s = s.replace(old_gate, '        if _CAP_N > 0 or _ROUTER in ("trt", "ffi") or _LOGIT_RING > 0:\n')
# the Modular class must be forced for ffi as for trt (whatever mechanism the hook uses for trt)
n_trt = s.count('_ROUTER == "trt"')
open(p, "w").write(s)
print("ok; remaining _ROUTER==trt sites:", n_trt)
