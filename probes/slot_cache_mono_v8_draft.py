"""slot_cache_mono.py — hot-expert slot cache, Monolithic edition (v8).

Design (proven offline on 3041 real layer-3 tokens, 0 mismatches at M=1 and M=2 vs Monolithic-256):
  * keep vLLM's stock Monolithic class (in-kernel DeepSeekV3 routing) — no Modular force, no external router, no packing.
  * per layer: S slots in HBM holding kernel-format [w13, w2, w13_scale, w2_scale] rows + per-expert scalars; all 256
    experts stay exact-pinned in host (UVA views) as the miss source.
  * SORTED slot layout: slot j holds the j-th smallest resident expert id. The kernel's tie-break (lower index wins on
    equal score) is then identical to the 256-expert case. Empties sort last and get logit/bias = -1e4.
  * per call: route in slot space by REMAPPING router logits + bias + scalars with one gather each, then call
    trtllm_fp4_block_scale_moe(num_experts=S) on the slot tables. Same routing code/bits as production Monolithic.
  * fill: NoAuxTc gives the ids cheaply (4us; only ids are used, never its weights). Misses evict LRU, copy rows over
    C2C (Triton), then a sorted-insert permutation keeps the table ordered. All device-side, graph-safe (fixed shapes).
Env: SLOT_CACHE=<S> (0=off), SLOT_CACHE_BYPASS_TOKENS (M above this -> stock path over UVA), SLOT_CACHE_STATS_SEC.
"""
import os, sys, time, threading, torch

S_SLOTS = int(os.environ.get("SLOT_CACHE", "0"))
BYPASS_ABOVE = int(os.environ.get("SLOT_CACHE_BYPASS_TOKENS", "16"))
STATS_SEC = int(os.environ.get("SLOT_CACHE_STATS_SEC", "20"))
NEG = -1e4

def _LOG(m): print(f"SLOT_CACHE_MONO {m}", file=sys.stderr, flush=True)

_registry = {}      # w1.data_ptr() -> LayerCache
_triton = None

def _install_triton():
    import triton, triton.language as tl

    @triton.jit
    def lru_plan(ids_ptr, e2s_ptr, s2e_ptr, last_ptr, step_ptr, src_out_ptr, dst_out_ptr, mask_out_ptr, miss_count_ptr,
                 N: tl.constexpr, S: tl.constexpr, E: tl.constexpr, SB: tl.constexpr):
        # single program. For each requested id: hit -> touch; miss -> pick LRU victim (not touched this call), assign.
        step = tl.load(step_ptr)
        soff = tl.arange(0, SB); smask = soff < S
        last = tl.load(last_ptr + soff, mask=smask, other=(1 << 62))
        for k in tl.static_range(N):                       # protect this call's hits from eviction
            e = tl.load(ids_ptr + k); sl = tl.load(e2s_ptr + e)
            last = tl.where((soff == sl) & (sl >= 0), (1 << 62), last)
        nmiss = 0
        for k in tl.static_range(N):
            e = tl.load(ids_ptr + k); sl = tl.load(e2s_ptr + e)
            seen = tl.zeros((), dtype=tl.int1)
            for j in tl.static_range(N):
                if j < k:
                    seen = seen | (tl.load(ids_ptr + j) == e)
            is_miss = (sl < 0) & (seen == 0)
            vmin = tl.min(last, 0)
            victim = tl.min(tl.where(last == vmin, soff, SB), 0)
            dst = tl.where(is_miss, victim, S)
            old_e = tl.load(s2e_ptr + dst)
            if is_miss:
                tl.store(e2s_ptr + old_e, -1)
                tl.store(e2s_ptr + e, dst)
                tl.store(s2e_ptr + dst, e)
                last = tl.where(soff == dst, (1 << 62), last)
                nmiss += 1
            tl.debug_barrier()
            sl2 = tl.load(e2s_ptr + e)
            final = tl.where(is_miss, dst, sl2)
            tl.store(src_out_ptr + k, tl.where(is_miss, e, E))
            tl.store(dst_out_ptr + k, dst)
            tl.store(mask_out_ptr + k, is_miss.to(tl.int8))
            tl.store(last_ptr + final, step, mask=final < S)
        tl.store(step_ptr, step + 1)
        tl.atomic_add(miss_count_ptr, nmiss)

    @triton.jit
    def masked_row_copy(src_ptr, dst_ptr, src_idx_ptr, dst_idx_ptr, mask_ptr, row_elems, BLOCK: tl.constexpr):
        k = tl.program_id(0); blk = tl.program_id(1)
        m = tl.load(mask_ptr + k)
        s = tl.load(src_idx_ptr + k).to(tl.int64); d = tl.load(dst_idx_ptr + k).to(tl.int64)
        off = blk * BLOCK + tl.arange(0, BLOCK)
        valid = (off < row_elems) & (m != 0)
        v = tl.load(src_ptr + s * row_elems + off, mask=valid, other=0)
        tl.store(dst_ptr + d * row_elems + off, v, mask=valid)

    @triton.jit
    def perm_rows(src_ptr, dst_ptr, perm_ptr, row_elems, BLOCK: tl.constexpr):
        # dst[j] = src[perm[j]]  (full table permutation into a scratch table)
        j = tl.program_id(0); blk = tl.program_id(1)
        p = tl.load(perm_ptr + j).to(tl.int64)
        off = blk * BLOCK + tl.arange(0, BLOCK); valid = off < row_elems
        v = tl.load(src_ptr + p * row_elems + off, mask=valid, other=0)
        tl.store(dst_ptr + j.to(tl.int64) * row_elems + off, v, mask=valid)

    return triton, lru_plan, masked_row_copy, perm_rows

def _ensure_triton():
    global _triton
    if _triton is None: _triton = _install_triton()
    return _triton


class LayerCache:
    def __init__(self, name, w13, w2, w13_scale, w2_scale, scalars, S):
        self.name, self.S, self.E = name, S, w13.shape[0]
        dev = torch.device("cuda"); self.dev = dev
        self.host = {"w13": w13, "w2": w2}
        # resident per-expert scales moved to pinned host (UVA), same as v6; slot copies in HBM
        from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor
        self.res, self._keep_host = {}, []
        for k, v in (("w13_scale", w13_scale), ("w2_scale", w2_scale)):
            if isinstance(v, torch.nn.Parameter) and v.is_cuda and os.environ.get("SLOT_CACHE_SCALES_TO_HOST", "1") == "1":
                try:
                    import exact_pin as _ep; host = _ep.exact_pinned_like(v.detach().to("cpu"))
                except Exception:
                    host = v.detach().to("cpu").pin_memory()
                self._keep_host.append(host); v.data = get_accelerator_view_from_cpu_tensor(host); self.res[k] = v.data
            else:
                self.res[k] = v.data if isinstance(v, torch.nn.Parameter) else v
        # slot tables: two copies (A/B) so a sorted permutation can be applied out-of-place then swapped
        def mk(t): return torch.empty((S,) + tuple(t.shape[1:]), dtype=t.dtype, device=dev)
        self.tab = [{"w13": mk(w13), "w2": mk(w2), "w13_scale": mk(self.res["w13_scale"]), "w2_scale": mk(self.res["w2_scale"])} for _ in range(2)]
        self.cur = 0
        self.scalars = [s for s in scalars]                                   # [E] fp32 each (g1c, g1a, g2a); may be None
        self.e2s = torch.full((self.E + 1,), -1, dtype=torch.int32, device=dev)
        self.s2e = torch.full((S + 1,), self.E, dtype=torch.int32, device=dev)
        self.last = torch.full((S,), -1, dtype=torch.int64, device=dev)
        self.step = torch.zeros((), dtype=torch.int64, device=dev)
        self.misses = torch.zeros((), dtype=torch.int64, device=dev)
        self.calls = torch.zeros((), dtype=torch.int64, device=dev)
        self.SB = 1
        while self.SB < S: self.SB *= 2
        self.rows = {k: (self._rows64(self.host[k] if k in self.host else self.res[k]), [self._rows64(t[k]) for t in self.tab]) for k in self.tab[0]}
        self.bufs = {}
        # remap buffers (fixed shapes; graph-safe)
        self.idx = torch.zeros(S, dtype=torch.long, device=dev)               # slot -> expert id (E for empty), clamped for gathers
        self.valid = torch.zeros(S, dtype=torch.bool, device=dev)
        self.bias_slot = torch.full((S,), NEG, dtype=torch.float32, device=dev)
        self.scal_slot = [torch.zeros(S, dtype=torch.float32, device=dev) if s is not None else None for s in self.scalars]
        self.perm = torch.arange(S, dtype=torch.int32, device=dev)
        self.perm64 = torch.arange(S, dtype=torch.long, device=dev)
        self.arangeS = torch.arange(S, device=dev)

    @staticmethod
    def _rows64(t):
        n = t.shape[0]; flat = t.contiguous().view(torch.uint8); return flat.view(torch.int64).reshape(n, -1)

    def bufs_for(self, N):
        if N not in self.bufs:
            d = self.dev
            self.bufs[N] = dict(src=torch.zeros(N, dtype=torch.int32, device=d), dst=torch.zeros(N, dtype=torch.int32, device=d),
                                mask=torch.zeros(N, dtype=torch.int8, device=d))
        return self.bufs[N]

    def tables(self): return self.tab[self.cur]

    def fill(self, ids_flat, N):
        """LRU plan + row copies for misses + keep table sorted by expert id. Device-side only."""
        triton, plan, mcopy, prows = _ensure_triton()
        b = self.bufs_for(N)
        plan[(1,)](ids_flat, self.e2s, self.s2e, self.last, self.step, b["src"], b["dst"], b["mask"], self.misses,
                   N=N, S=self.S, E=self.E, SB=self.SB)
        BLOCK = 2048; T = self.tables()
        for k in T:
            src = self.rows[k][0]; dst = self.rows[k][1][self.cur]; n = src.shape[1]
            mcopy[(N, triton.cdiv(n, BLOCK))](src, dst, b["src"], b["dst"], b["mask"], n, BLOCK=BLOCK)
        # sorted-insert: compute permutation (residents by id, empties last) and apply into the other table copy.
        # Always executed (fixed work; ~S row copies of HBM->HBM ≈ 2 GB/layer at ~6 TB/s ≈ 0.35 ms) — see NOTE below.
        s2e = self.s2e[: self.S]
        key = torch.where(s2e < self.E, s2e, torch.full_like(s2e, self.E)).to(torch.int64) * self.S + self.arangeS   # stable
        perm = torch.argsort(key)
        self.perm64.copy_(perm); self.perm.copy_(perm.to(torch.int32))
        nxt = 1 - self.cur
        for k in T:
            src = self.rows[k][1][self.cur]; dst = self.rows[k][1][nxt]; n = src.shape[1]
            prows[(self.S, triton.cdiv(n, BLOCK))](src, dst, self.perm, n, BLOCK=BLOCK)
        # permute metadata
        new_s2e = s2e[perm]; self.s2e[: self.S].copy_(new_s2e); self.last.copy_(self.last[perm])
        res = new_s2e < self.E
        self.e2s.fill_(-1); self.e2s[new_s2e.long()] = torch.where(res, self.arangeS.to(torch.int32), torch.full_like(self.arangeS, -1, dtype=torch.int32))
        self.e2s[self.E] = -1
        self.cur = nxt
        # remap vectors for this layout
        self.valid.copy_(res); self.idx.copy_(torch.where(res, new_s2e.long(), torch.zeros_like(new_s2e.long())))
        self.calls += 1

    def remap(self, logits, bias):
        lgS = logits[:, self.idx]; lgS = torch.where(self.valid[None, :], lgS, torch.full_like(lgS, NEG))
        bS = torch.where(self.valid, bias[self.idx].float(), self.bias_slot)
        sc = [s[self.idx] if s is not None else None for s in self.scalars]
        return lgS.contiguous(), bS.contiguous(), sc


def install():
    if S_SLOTS <= 0: return
    import importlib.abc, importlib.util

    def _patch_uva(module):
        cls = getattr(module, "UVAOffloader", None) or next(c for c in vars(module).values() if isinstance(c, type) and hasattr(c, "_maybe_offload_to_cpu"))
        orig = cls._maybe_offload_to_cpu
        def _maybe_offload_to_cpu(self, mod):
            r = orig(self, mod)
            for name, sub in mod.named_modules():
                w13 = getattr(sub, "w13_weight", None); w2 = getattr(sub, "w2_weight", None)
                if w13 is None or w2 is None: continue
                if getattr(w13, "_vllm_is_uva_offloaded", False) and getattr(w2, "_vllm_is_uva_offloaded", False):
                    sub._slot_cache_pending = True
            return r
        cls._maybe_offload_to_cpu = _maybe_offload_to_cpu
        _LOG("uva offloader hook installed")

    def _patch_experts(module):
        Mono = module.TrtLlmNvFp4ExpertsMonolithic
        Base = module.TrtLlmNvFp4ExpertsBase
        orig_pwal = Base.process_weights_after_loading
        def process_weights_after_loading(self, layer):
            orig_pwal(self, layer)
            if getattr(layer, "_slot_cache_pending", False):
                qc = self.quant_config
                w1s = getattr(layer, "w13_weight_scale", qc.w1_scale); w2s = getattr(layer, "w2_weight_scale", qc.w2_scale)
                self._slot_cache_deferred = getattr(self, "_slot_cache_deferred", {})
                self._slot_cache_deferred[id(layer)] = (getattr(layer, "layer_name", "?"), w1s, w2s, [self.g1_scale_c, qc.g1_alphas, qc.g2_alphas])
        Base.process_weights_after_loading = process_weights_after_loading

        import flashinfer
        from flashinfer.fused_moe.fused_routing_dsv3 import get_dsv3_fused_routing_module
        _route = get_dsv3_fused_routing_module()
        orig_apply = Mono.apply
        def apply(self, hidden_states, w1, w2, router_logits, activation, global_num_experts, expert_map, a1q_scale,
                  apply_router_weight_on_input, num_expert_group=None, e_score_correction_bias=None, routed_scaling_factor=None, topk_group=None):
            lc = _registry.get(w1.data_ptr())
            if lc is None and getattr(self, "_slot_cache_deferred", None):
                for key, (name, w1s, w2s, scalars) in list(self._slot_cache_deferred.items()):
                    if w1s.data.data_ptr() == self.quant_config.w1_scale.data_ptr():
                        if not w1.is_cuda: break
                        lc = LayerCache(name, w1, w2, w1s, w2s, scalars, S_SLOTS); _registry[w1.data_ptr()] = lc
                        del self._slot_cache_deferred[key]
                        _LOG(f"cache built for {name}: S={S_SLOTS} E={lc.E} slot bytes/copy={sum(t.numel()*t.element_size() for t in lc.tab[0].values())/1e9:.2f} GB x2")
                        if len(_registry) >= 75: _start_stats_thread()
                        break
            M = hidden_states.shape[0]; K = self.topk
            if lc is None or M > BYPASS_ABOVE or M * K > lc.S:
                return orig_apply(self, hidden_states, w1, w2, router_logits, activation, global_num_experts, expert_map, a1q_scale,
                                  apply_router_weight_on_input, num_expert_group, e_score_correction_bias, routed_scaling_factor, topk_group)
            # 1) ids via NoAuxTc (ids only; matches in-kernel selection 3041/3041 incl. ties)
            lg32 = router_logits.float().contiguous(); bias32 = e_score_correction_bias.float().contiguous()
            tv = torch.empty(M, K, dtype=torch.float32, device=lg32.device); ti = torch.empty(M, K, dtype=torch.int32, device=lg32.device)
            _route.NoAuxTc(lg32, bias32, 1, 1, K, 1.0, tv, ti, False, None)
            # 2) fill misses + keep sorted
            lc.fill(ti.reshape(-1), M * K)
            # 3) remap into slot space and run Monolithic on the slot tables
            lgS, bS, (g1c, g1a, g2a) = lc.remap(router_logits, e_score_correction_bias)
            T = lc.tables()
            block_scale = a1q_scale
            result = flashinfer.fused_moe.trtllm_fp4_block_scale_moe(
                routing_logits=lgS.to(router_logits.dtype), routing_bias=bS.to(e_score_correction_bias.dtype),
                hidden_states=hidden_states, hidden_states_scale=block_scale.view(torch.float8_e4m3fn).reshape(*hidden_states.shape[:-1], -1),
                gemm1_weights=T["w13"], gemm1_weights_scale=T["w13_scale"].view(torch.float8_e4m3fn), gemm1_bias=None,
                gemm1_alpha=self.gemm1_alpha, gemm1_beta=self.gemm1_beta, gemm1_clamp_limit=self.gemm1_clamp_limit,
                gemm2_weights=T["w2"], gemm2_weights_scale=T["w2_scale"].view(torch.float8_e4m3fn), gemm2_bias=None,
                output1_scale_scalar=g1c, output1_scale_gate_scalar=g1a, output2_scale_scalar=g2a,
                num_experts=lc.S, top_k=K, n_group=(num_expert_group or 0), topk_group=(topk_group or 0),
                intermediate_size=self.intermediate_size_per_partition, local_expert_offset=0, local_num_experts=lc.S,
                routed_scaling_factor=routed_scaling_factor, routing_method_type=self.routing_method_type, do_finalize=True,
                activation_type=module.activation_to_flashinfer_int(activation), per_token_scale=None,
                tune_max_num_tokens=lc.S, routing_replay_out=None)
            return result[0]
        Mono.apply = apply
        _LOG(f"Monolithic slot path installed: S={S_SLOTS} bypass_above_tokens={BYPASS_ABOVE}")

    targets = {"vllm.model_executor.offloader.uva": _patch_uva,
               "vllm.model_executor.layers.fused_moe.experts.trtllm_nvfp4_moe": _patch_experts}

    class _Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path, target=None):
            if name not in targets: return None
            fn = targets.pop(name)
            if not targets:
                try: sys.meta_path.remove(self)
                except ValueError: pass
            spec = importlib.util.find_spec(name)
            if spec is None or spec.loader is None: return None
            loader = spec.loader; orig_exec = loader.exec_module
            def exec_module(mod):
                orig_exec(mod)
                try: fn(mod)
                except Exception as e: _LOG(f"patch {name} failed: {e!r}")
            loader.exec_module = exec_module
            return spec
    sys.meta_path.insert(0, _Finder())
    _LOG(f"armed: S={S_SLOTS} bypass_above={BYPASS_ABOVE}")


_stats_thread = None
def _start_stats_thread():
    global _stats_thread
    if _stats_thread is not None or STATS_SEC <= 0: return
    def run():
        prev = {}
        while True:
            time.sleep(STATS_SEC)
            try:
                tot_m = 0; tot_c = 0; per = []
                for lc in list(_registry.values()):
                    m = int(lc.misses.item()); c = int(lc.calls.item()); tot_m += m; tot_c += c
                    pm, pc = prev.get(lc.name, (0, 0)); dm, dc = m - pm, c - pc; prev[lc.name] = (m, c)
                    if dc > 0: per.append((lc.name, dm / max(1, dc * 8)))
                if per:
                    per.sort(key=lambda x: -x[1])
                    _LOG(f"stats calls={tot_c} misses={tot_m} worst_miss_rate={per[0][0]}:{per[0][1]:.3f} best={per[-1][0]}:{per[-1][1]:.3f} mean={sum(p[1] for p in per)/len(per):.3f}")
            except Exception as e:
                _LOG(f"stats error {e!r}")
    _stats_thread = threading.Thread(target=run, daemon=True, name="slotcache-stats"); _stats_thread.start()

install()
