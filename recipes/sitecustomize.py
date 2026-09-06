# sitecustomize.py — bind-mounted over /usr/lib/python3.12/sitecustomize.py in the vLLM container.
# Keeps Ubuntu's apport hook, then installs a lazy import hook that patches
# RoutedExpertsManager.store_batch to append every step's routing tensor to disk.
# Activated only when ROUTE_TRACE_DIR is set. Zero effect on the compute graph.
try:
    import apport_python_hook
except ImportError:
    pass
else:
    apport_python_hook.install()

import os, sys
_OUT = os.environ.get("ROUTE_TRACE_DIR")
_TARGET = "vllm.model_executor.layers.fused_moe.routed_experts_capturer"
_AT_KEY = os.environ.get("VLLM_AUTOTUNE_CACHE_KEY")  # pin the FlashInfer autotune cache dir by name
_AT_TARGET = "vllm.model_executor.warmup.flashinfer_autotune_cache"

if _AT_KEY:
    import importlib.abc, importlib.util

    def _patch_at(module):
        module.flashinfer_autotune_cache_hash = lambda runner: _AT_KEY
        sys.stderr.write(f"AUTOTUNE_KEY hook installed: {_AT_KEY}\n")

    class _ATFinder(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path, target=None):
            if name != _AT_TARGET:
                return None
            sys.meta_path.remove(self)
            spec = importlib.util.find_spec(name)
            if spec is None or spec.loader is None:
                return None
            loader = spec.loader
            orig_exec = loader.exec_module

            def exec_module(module, _orig=orig_exec):
                _orig(module)
                _patch_at(module)

            loader.exec_module = exec_module
            return spec

    sys.meta_path.insert(0, _ATFinder())



# --- exact-size pinned host allocations (cudaHostAlloc) instead of torch.pin_memory() power-of-two rounding.
if os.environ.get("EXACT_PIN", "0") == "1":
    try:
        import importlib.util as _ilu2
        _sp2 = _ilu2.spec_from_file_location("exact_pin", os.environ.get("EXACT_PIN_FILE", "/w/exact_pin.py"))
        _ep = _ilu2.module_from_spec(_sp2); _sp2.loader.exec_module(_ep); _ep.install_uva_patch()
        sys.stderr.write("EXACT_PIN hook installed\n")
    except Exception as _e:
        sys.stderr.write(f"EXACT_PIN hook FAILED: {_e!r}\n")

# --- hot-expert slot cache (design v2). Activated by SLOT_CACHE=<slots>; hook file bind-mounted alongside.
if os.environ.get("SLOT_CACHE"):
    _hook_path = os.environ.get("SLOT_CACHE_HOOK", "/w/slot_cache_hook.py")
    try:
        import importlib.util as _ilu
        _spec = _ilu.spec_from_file_location("slot_cache_hook", _hook_path)
        _m = _ilu.module_from_spec(_spec); _spec.loader.exec_module(_m)
        _m.install()
    except Exception as _e:
        sys.stderr.write(f"SLOT_CACHE hook FAILED to install: {_e!r}\n")

# --- per-layer output trace for teacher-forced comparisons. Activated by LAYER_TRACE=1; hook file bind-mounted at /w.
if os.environ.get("LAYER_TRACE") == "1":
    import importlib.abc as _lt_abc, importlib.util as _lt_util
    _LT_TARGET = "vllm.model_executor.model_loader.base_loader"
    class _LTFinder(_lt_abc.MetaPathFinder):
        def find_spec(self, name, path, target=None):
            if name != _LT_TARGET:
                return None
            sys.meta_path.remove(self)
            spec = _lt_util.find_spec(name)
            if spec is None or spec.loader is None:
                return None
            loader = spec.loader
            orig_exec = loader.exec_module
            def exec_module(module, _orig=orig_exec):
                _orig(module)
                try:
                    _sp = _lt_util.spec_from_file_location("layer_trace", os.environ.get("LAYER_TRACE_FILE", "/w/layer_trace.py"))
                    _lt = _lt_util.module_from_spec(_sp); _sp.loader.exec_module(_lt); _lt.install()
                except Exception as _e:
                    sys.stderr.write(f"LAYER_TRACE hook FAILED: {_e!r}\n")
            loader.exec_module = exec_module
            return spec
    sys.meta_path.insert(0, _LTFinder())

if _OUT:
    import importlib.abc, importlib.util

    def _patch(module):
        import numpy as np, time
        # --- fix 1: get_routed_experts_attn_gid rejects UniformTypeKVCacheSpecs wrappers even when
        # every inner spec is a FullAttentionSpec (MLA models on TP1 land here). Unwrap.
        from vllm.v1.kv_cache_interface import FullAttentionSpec, UniformTypeKVCacheSpecs
        def _gid(kv_cache_config):
            for gid, group in enumerate(kv_cache_config.kv_cache_groups):
                spec = group.kv_cache_spec
                if isinstance(spec, FullAttentionSpec):
                    return gid
                if isinstance(spec, UniformTypeKVCacheSpecs) and all(
                    isinstance(s, FullAttentionSpec) for s in spec.kv_cache_specs.values()
                ):
                    return gid
            raise ValueError("Routed-experts capture requires a full-attention KV cache group.")
        module.get_routed_experts_attn_gid = _gid
        sys.stderr.write("ROUTE_TRACE attn_gid unwrap patch installed\n")
        # --- trace hook
        cls = module.RoutedExpertsManager
        orig = cls.store_batch
        os.makedirs(_OUT, exist_ok=True)
        pid = os.getpid()
        fb = open(os.path.join(_OUT, f"trace-{pid}.i16"), "ab")
        fs = open(os.path.join(_OUT, f"trace-{pid}.slots.i64"), "ab")
        fm = open(os.path.join(_OUT, f"trace-{pid}.steps"), "a")
        st = {"n": 0}

        def store_batch(self, data, slot_mapping):
            try:
                d = np.ascontiguousarray(data, dtype=np.int16)
                s = np.ascontiguousarray(slot_mapping, dtype=np.int64)
                fb.write(d.tobytes()); fs.write(s.tobytes())
                fm.write(f"{time.time():.3f} {d.shape[0]} {d.shape[1]} {d.shape[2]}\n")
                st["n"] += d.shape[0]
                if st["n"] % 2000 < d.shape[0]:
                    fb.flush(); fs.flush(); fm.flush()
            except Exception as e:  # never break serving
                sys.stderr.write(f"ROUTE_TRACE error: {e!r}\n")
            return orig(self, data, slot_mapping)

        cls.store_batch = store_batch
        sys.stderr.write(f"ROUTE_TRACE hook installed pid={pid} dir={_OUT}\n")

    class _Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path, target=None):
            if name != _TARGET:
                return None
            sys.meta_path.remove(self)
            spec = importlib.util.find_spec(name)
            if spec is None or spec.loader is None:
                return None
            loader = spec.loader
            orig_exec = loader.exec_module

            def exec_module(module, _orig=orig_exec):
                _orig(module)
                _patch(module)

            loader.exec_module = exec_module
            return spec

    sys.meta_path.insert(0, _Finder())
