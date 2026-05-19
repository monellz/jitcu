import shutil
import subprocess
from ctypes import CDLL, POINTER, Structure, byref, c_int32, c_int64, c_void_p
from enum import Enum
from pathlib import Path

import torch


def _exported_functions(lib_path: str) -> list[str] | None:
    """Best-effort list of `extern "C"` function names exported by a `.so`.

    Reads the ELF dynamic symbol table via `nm`; mangled C++ symbols (`_Z...`)
    are dropped so the result is just the C-linkage entrypoints. Returns
    ``None`` when `nm` is unavailable, so callers can degrade gracefully.
    """
    nm = shutil.which("nm")
    if nm is None:
        return None
    try:
        out = subprocess.run(
            [nm, "-D", "--defined-only", lib_path],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    except (subprocess.SubprocessError, OSError):
        return None
    funcs = []
    for line in out.splitlines():
        # nm -D lines look like `<addr> <type> <name>`; T/W are text symbols.
        parts = line.split()
        if len(parts) >= 3 and parts[-2] in ("T", "W") and not parts[-1].startswith("_Z"):
            funcs.append(parts[-1])
    return funcs


class Tensor(Structure):
    _fields_ = [
        ("data", c_void_p),
        ("ndim", c_int32),
        ("shape", POINTER(c_int64)),
        ("strides", POINTER(c_int64)),
        ("dtype", c_int32),
    ]

    class Dtype(Enum):
        UINT64 = 0
        INT64 = 1
        FLOAT64 = 2
        UINT32 = 3
        INT32 = 4
        FLOAT32 = 5
        UINT16 = 6
        INT16 = 7
        FLOAT16 = 8
        BFLOAT16 = 9
        UINT8 = 10
        INT8 = 11
        FLOAT8_E4M3FN = 12
        FLOAT8_E4M3FNUZ = 13
        FLOAT8_E5M2 = 14
        FLOAT8_E5M2FNUZ = 15

        @classmethod
        def from_torch_dtype(cls, dtype: torch.dtype):
            return {
                torch.uint64: cls.UINT64,
                torch.int64: cls.INT64,
                torch.float64: cls.FLOAT64,
                torch.uint32: cls.UINT32,
                torch.int32: cls.INT32,
                torch.float32: cls.FLOAT32,
                torch.uint16: cls.UINT16,
                torch.int16: cls.INT16,
                torch.float16: cls.FLOAT16,
                torch.bfloat16: cls.BFLOAT16,
                torch.uint8: cls.UINT8,
                torch.int8: cls.INT8,
                torch.float8_e4m3fn: cls.FLOAT8_E4M3FN,
                torch.float8_e4m3fnuz: cls.FLOAT8_E4M3FNUZ,
                torch.float8_e5m2: cls.FLOAT8_E5M2,
                torch.float8_e5m2fnuz: cls.FLOAT8_E5M2FNUZ,
            }[dtype]

    @classmethod
    def from_torch_tensor(cls, src: torch.Tensor):
        assert src.storage_offset() == 0
        data = c_void_p(src.data_ptr())
        ndim = src.dim()
        dtype = cls.Dtype.from_torch_dtype(src.dtype).value
        shape = (c_int64 * ndim)(*src.shape)
        strides = (c_int64 * ndim)(*src.stride())
        return cls(
            data=data,
            ndim=ndim,
            dtype=dtype,
            shape=shape,
            strides=strides,
        )


class Library:
    type_mapping = {
        "t": POINTER(Tensor),
        "i32": c_int32,
        "i64": c_int64,
    }

    def __init__(
        self,
        lib_path: str | Path,
        func_specs: dict[str, str],
        device_type: str = "cuda",
    ):
        self.lib_path = str(lib_path)
        self.func_specs = func_specs
        self.device_type = device_type
        assert self.device_type in [
            "cuda",
            "npu",
        ], f"Unsupported device type: {self.device_type}"

        self._load()

    def _load(self):
        self.lib = CDLL(self.lib_path)
        for func_name, func_param in self.func_specs.items():
            try:
                func = getattr(self.lib, func_name)
            except AttributeError:
                exported = _exported_functions(self.lib_path)
                hint = (
                    f" Exported functions: {exported}"
                    if exported is not None
                    else ""
                )
                raise RuntimeError(
                    f"function {func_name!r} not found in {self.lib_path}.{hint}"
                ) from None
            # arg 0 of called function is always the cuda stream
            func.argtypes = [c_void_p] + [
                self.type_mapping[p]
                for p in filter(
                    lambda x: len(x.strip()) > 0, func_param.strip().split("_")
                )
            ]
            func.restype = None

            def make_wrapper(func, device_type):
                if device_type == "cuda":

                    def _func(*args):
                        stream = torch.cuda.current_stream().cuda_stream
                        args = [
                            (
                                byref(Tensor.from_torch_tensor(arg))
                                if isinstance(arg, torch.Tensor)
                                else arg
                            )
                            for arg in args
                        ]
                        return func(stream, *args)

                elif device_type == "npu":

                    def _func(*args):
                        stream = torch.npu.current_stream().npu_stream  # ty: ignore[unresolved-attribute]
                        args = [
                            (
                                byref(Tensor.from_torch_tensor(arg))
                                if isinstance(arg, torch.Tensor)
                                else arg
                            )
                            for arg in args
                        ]
                        return func(stream, *args)

                else:
                    raise NotImplementedError(f"Unsupported device type: {device_type}")
                return _func

            self.__setattr__(func_name, make_wrapper(func, self.device_type))
