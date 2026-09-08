"""Exercise the real ctypes ABI with a host library; no accelerator required."""

import math
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from jitcu.library import Library


@pytest.fixture(scope="module")
def scalar_library(tmp_path_factory):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("scalar ABI test requires a host C++ compiler")
    directory = tmp_path_factory.mktemp("jitcu_scalar_abi")
    source = directory / "scalars.cpp"
    source.write_text(
        r"""
#include "jitcu/tensor.h"

extern "C" void scalar_args(void *stream, jc::Tensor &out,
                            int32_t i32, float f32, int64_t i64, double f64) {
    auto *values = out.data_ptr<double>();
    values[0] = i32;
    values[1] = f32;
    values[2] = i64;
    values[3] = f64;
    values[4] = reinterpret_cast<uintptr_t>(stream);
}
"""
    )
    library = directory / "scalars.so"
    include = Path(__file__).resolve().parents[1] / "jitcu" / "data" / "include"
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-shared",
            "-fPIC",
            f"-I{include}",
            str(source),
            "-o",
            str(library),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return library


@pytest.mark.parametrize("device_type", ["cuda", "npu"])
@pytest.mark.parametrize(
    "i32,f32,i64,f64",
    [
        (17, 0.1, 2**40 + 3, 1 + 2**-40),
        (-9, -1e-6, -(2**40) + 7, -1e-100),
        (0, -0.0, 0, -0.0),
        (1, float("inf"), 2, float("nan")),
    ],
)
def test_scalar_arguments(scalar_library, monkeypatch, device_type, i32, f32, i64, f64):
    stream_address = 2**40 + 5
    stream = SimpleNamespace(**{f"{device_type}_stream": stream_address})
    backend = getattr(torch, device_type, None)
    if backend is None:
        backend = SimpleNamespace()
        monkeypatch.setattr(torch, device_type, backend, raising=False)
    monkeypatch.setattr(backend, "current_stream", lambda: stream, raising=False)
    lib = Library(
        scalar_library,
        {"scalar_args": "t_i32_f32_i64_f64"},
        device_type=device_type,
    )
    output = torch.empty(5, dtype=torch.float64, device="cpu")
    lib.scalar_args(output, i32, f32, i64, f64)
    expected = torch.tensor(
        [i32, torch.tensor(f32, dtype=torch.float32).item(), i64, f64, stream_address],
        dtype=torch.float64,
        device="cpu",
    )
    torch.testing.assert_close(output, expected, atol=0, rtol=0, equal_nan=True)
    for index, value in ((1, f32), (3, f64)):
        if value == 0:
            assert math.copysign(1, output[index].item()) == math.copysign(1, value)
