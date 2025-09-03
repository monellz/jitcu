import os
from typing import Callable

import torch
import triton


def benchmark(
    tag: str,
    fn: Callable,
    fn_ref: Callable,
    verify_fn: Callable = None,
    flops: int = None,
    byte: int = None,
    return_type: str = "compute",
):
    assert return_type in ["compute", "memory", "time"], f"{return_type=}"

    verify = os.environ.get("verify", "0") == "1"
    once = os.environ.get("once", "0") == "1"

    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    out = fn()
    end.record()
    torch.cuda.synchronize()

    ms = start.elapsed_time(end)

    if verify:
        out_ref = fn_ref()
        if verify_fn is not None:
            verify_fn(out, out_ref)
        else:
            assert out.isfinite().all(), f"{out=}"
            assert out_ref.isfinite().all(), f"{out_ref=}"
            torch.testing.assert_close(out, out_ref)

    if not once:
        torch.cuda.synchronize()
        with torch.cuda.stream(torch.cuda.Stream()):
            ms = triton.testing.do_bench_cudagraph(fn)

    ai = flops / byte

    msg = f"{tag}: {ms} ms"
    if flops is not None:
        tflops_s = flops / 1e12 / (ms / 1000)
        msg += f", {tflops_s} TFLOPS/s"
    if byte is not None:
        gb_s = byte / 1e9 / (ms / 1000)
        msg += f", {gb_s} GB/s"
    if flops is not None and byte is not None:
        msg += f", AI {ai}"
    msg += f" ({verify=}, {once=})"

    print(msg, flush=True)

    if return_type == "compute":
        return tflops_s
    elif return_type == "memory":
        return gb_s
    elif return_type == "time":
        return ms
