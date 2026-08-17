"""Device-agnostic graph-replay benchmark.

`do_bench_graph` is the analog of `triton.testing.do_bench_cudagraph`, but:
  - works on any torch device backend that exposes a graph type (CUDA -> CUDAGraph,
    Ascend NPU -> NPUGraph), resolved by name rather than hard-coded;
  - optionally flushes a >L2 cache buffer before each timed replay so the kernel reads
    cold from HBM/global memory (do_bench_cudagraph is L2-warm by design).

Capturing the launch into a graph and timing replay() removes ~all host launch overhead,
so the result is essentially the device time — useful when the kernel is small enough that
the host/ctypes dispatch dominates a plain wall-clock timer.
"""

import torch


def _resolve_device(device_type):
    """Return (device_type, device_module, graph_cls) for cuda / npu / ... by name."""
    if device_type is None:
        if torch.cuda.is_available():
            device_type = "cuda"
        elif hasattr(torch, "npu") and torch.npu.is_available():
            device_type = "npu"
        else:
            raise RuntimeError("do_bench_graph: no cuda/npu device found; pass device_type=")
    mod = getattr(torch, device_type, None)
    if mod is None:
        raise RuntimeError(f"do_bench_graph: torch.{device_type} not available")
    # torch.cuda.CUDAGraph, torch.npu.NPUGraph, ...
    graph_cls = getattr(mod, device_type.upper() + "Graph", None)
    if graph_cls is None:
        raise RuntimeError(f"do_bench_graph: torch.{device_type} exposes no graph type")
    return device_type, mod, graph_cls


def _summarize(times, quantiles, return_mode):
    t = torch.tensor(times, dtype=torch.float)
    if quantiles is not None:
        res = torch.quantile(t, torch.tensor(quantiles, dtype=torch.float)).tolist()
        return res[0] if len(res) == 1 else res
    if return_mode == "all":
        return t.tolist()
    return {"min": t.min(), "max": t.max(), "mean": t.mean(), "median": t.median()}[return_mode].item()


def do_bench_graph(
    fn,
    rep=20,
    warmup=5,
    flush=True,
    cache_size_mb=256,
    n_retries=10,
    quantiles=None,
    return_mode="median",
    device_type=None,
):
    """Benchmark `fn` via device-graph capture + replay (CUDA Graph / NPUGraph).

    :param fn: zero-arg callable that enqueues the work on the current stream.
    :param rep: target measurement window (ms); sizes how many calls are unrolled / repeated.
    :param warmup: number of plain fn() calls before capture.
    :param flush: if True, zero a >L2 cache buffer before each timed replay (cold reads); the
        flush is enqueued before the start event so it is NOT in the timing window. If False,
        mirror do_bench_cudagraph: capture N unrolled calls back-to-back (L2-warm, min host).
    :param cache_size_mb: size of the cache-eviction buffer (only used when flush=True).
    :param return_mode: "min" | "max" | "mean" | "median" | "all".
    :param quantiles: optional list of quantiles to return instead of return_mode.
    :param device_type: "cuda" | "npu" | None (auto-detect).
    :return: per-call time in ms (or a list for quantiles / return_mode="all").
    """
    assert return_mode in ("min", "max", "mean", "median", "all")
    dt, mod, graph_cls = _resolve_device(device_type)
    dev = f"{dt}:{mod.current_device()}"

    with mod.stream(mod.Stream()):
        for _ in range(warmup):
            fn()
        # estimate one call (no graph) to size the repeat count, like do_bench_cudagraph
        s = mod.Event(enable_timing=True)
        e = mod.Event(enable_timing=True)
        s.record()
        for _ in range(5):
            fn()
        e.record()
        mod.synchronize()
        est_ms = max(s.elapsed_time(e) / 5, 1e-4)
        n_repeat = max(1, int(rep / est_ms))

        if flush:
            # cold: capture a single call; evict the cache before each timed replay.
            cache = torch.empty(cache_size_mb * 1024 * 1024, dtype=torch.int8, device=dev)
            g = graph_cls()
            with mod.graph(g):
                fn()
            mod.synchronize()
            for _ in range(5):
                g.replay()
            mod.synchronize()
            times = []
            for _ in range(max(n_retries, n_repeat)):
                cache.zero_()  # enqueued before the start event -> evicts cache, not timed
                s = mod.Event(enable_timing=True)
                e = mod.Event(enable_timing=True)
                s.record()
                g.replay()
                e.record()
                mod.synchronize()
                times.append(s.elapsed_time(e))
        else:
            # warm: capture n_repeat unrolled calls; one replay -> n_repeat kernels.
            g = graph_cls()
            with mod.graph(g):
                for _ in range(n_repeat):
                    fn()
            mod.synchronize()
            times = []
            for _ in range(n_retries):
                s = mod.Event(enable_timing=True)
                e = mod.Event(enable_timing=True)
                s.record()
                g.replay()
                e.record()
                mod.synchronize()
                times.append(s.elapsed_time(e) / n_repeat)

    return _summarize(times, quantiles, return_mode)
