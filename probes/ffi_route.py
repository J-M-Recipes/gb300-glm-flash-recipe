"""ffi_route.py — call FlashInfer's Monolithic routing pass (Routing::Runner::run inside the already-loaded
fused_moe_trtllm_sm100.so) standalone via ctypes. Same compiled kernel Monolithic uses => same bf16 weight bits.
Returns the kernel's own packed (bf16 score | int16 idx) int32 [M,K], its bf16 weights [M,K], and int16 replay ids [M,K].
No FlashInfer source patch; no recompilation."""
import ctypes, torch

_SYM_RUN = "_ZN12tensorrt_llm7kernels13trtllmgen_moe7Routing6Runner3runEPvS4_iiiiiiiifPiS5_S5_S5_S5_S5_S5_S4_S5_S5_S5_S5_N11batchedGemm6trtllm3gen5DtypeES9_bbNS2_17RoutingMethodTypeEP11CUstream_stS9_bPsb"
_SYM_CTOR = "_ZN12tensorrt_llm7kernels13trtllmgen_moe7Routing6RunnerC1Ei"
# btg::Dtype encoding: (block<<24)|(signed<<20)|(int<<16)|(bits<<8)|uid
DT_BF16 = (1 << 20) | (16 << 8) | 0
DT_FP32 = (1 << 20) | (32 << 8) | 8
DT_E2M1 = (1 << 24) | (1 << 20) | (4 << 8) | 2
RM_DEEPSEEKV3 = 2

_lib = None; _run = None; _runner = None

def _load():
    global _lib, _run, _runner
    if _run is not None: return
    from flashinfer.fused_moe.core import get_trtllm_moe_sm100_module
    get_trtllm_moe_sm100_module()                                   # make sure the module is loaded/initialised in-process
    from flashinfer.jit.fused_moe import gen_trtllm_gen_fused_moe_sm100_module
    path = str(gen_trtllm_gen_fused_moe_sm100_module(enable_rubin=False).get_library_path())
    _lib = ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
    ctor = getattr(_lib, _SYM_CTOR); ctor.restype = None; ctor.argtypes = [ctypes.c_void_p, ctypes.c_int32]
    _runner = ctypes.create_string_buffer(64)                          # Runner{int32 mTileTokensDim} — oversized on purpose
    ctor(ctypes.addressof(_runner), 8)                                 # tile_tokens_dim=8 (Monolithic's choice for M*K/E < 1)
    _run = getattr(_lib, _SYM_RUN); _run.restype = None
    P = ctypes.c_void_p
    _run.argtypes = [P, P, P, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
                     ctypes.c_int32, ctypes.c_int32, ctypes.c_float,
                     P, P, P, P, P, P, P, P, P, P, P, P,
                     ctypes.c_uint32, ctypes.c_uint32, ctypes.c_bool, ctypes.c_bool, ctypes.c_int64, P, ctypes.c_uint32,
                     ctypes.c_bool, P, ctypes.c_bool]

def _max_ctas(M, K, E, tile):
    rem = M * K; filled = min(E, rem); ctas = filled; rem -= filled
    if rem > 0: ctas += rem // tile
    return ctas

class RouteWorkspace:
    """Fixed-shape scratch for a given (M, K, E); reusable across calls (graph-safe: no allocation per call)."""
    def __init__(self, M, K, E, device, tile=8):
        d = device
        self.M, self.K, self.E, self.tile = M, K, E, tile
        ctas = _max_ctas(M, K, E, tile); padded = ctas * tile
        self.packed = torch.zeros(M, K, dtype=torch.int32, device=d)           # routingExpertIndexes = PackedScoreIdx<bf16>
        self.weights = torch.zeros(M, K, dtype=torch.bfloat16, device=d)        # mPtrTopKWeights
        self.replay = torch.zeros(M, K, dtype=torch.int16, device=d)
        self.hist = torch.zeros(max(2 * E, 512), dtype=torch.int32, device=d)
        self.padded_size = torch.zeros(1, dtype=torch.int32, device=d)
        self.exp2perm = torch.zeros(M * K, dtype=torch.int32, device=d)
        self.perm2tok = torch.zeros(padded + 1, dtype=torch.int32, device=d)
        self.per_expert = torch.zeros(E, dtype=torch.int32, device=d)
        self.cta_batch = torch.zeros(max(ctas, 1), dtype=torch.int32, device=d)
        self.cta_mn = torch.zeros(max(ctas, 1), dtype=torch.int32, device=d)
        self.non_exiting = torch.zeros(1, dtype=torch.int32, device=d)

def mono_route(logits, bias, ws, routed_scaling_factor=2.5, n_group=1, topk_group=1, norm_topk_prob=True, enable_pdl=True):
    """logits [M,E] (bf16|fp32), bias [E] (bf16|fp32) — exactly what vLLM hands Monolithic. Writes ws.packed/ws.weights/ws.replay."""
    _load()
    assert logits.is_contiguous() and bias.is_contiguous()
    M, E = logits.shape; K = ws.K
    assert M == ws.M and E == ws.E
    dt_logits = DT_FP32 if logits.dtype == torch.float32 else DT_BF16
    dt_bias = DT_BF16 if bias.dtype == torch.bfloat16 else DT_FP32
    stream = torch.cuda.current_stream(logits.device).cuda_stream
    _run(ctypes.addressof(_runner), logits.data_ptr(), bias.data_ptr(), M, E, K, 0, n_group, topk_group, 0, E,
         float(routed_scaling_factor),
         ws.packed.data_ptr(), ws.hist.data_ptr(), ws.padded_size.data_ptr(), ws.exp2perm.data_ptr(), None,
         ws.perm2tok.data_ptr(), None, ws.weights.data_ptr(), ws.per_expert.data_ptr(), ws.cta_batch.data_ptr(),
         ws.cta_mn.data_ptr(), ws.non_exiting.data_ptr(),
         DT_E2M1, dt_bias, False, False, RM_DEEPSEEKV3, stream, dt_logits, bool(norm_topk_prob), ws.replay.data_ptr(),
         bool(enable_pdl))
    return ws.packed, ws.weights, ws.replay
