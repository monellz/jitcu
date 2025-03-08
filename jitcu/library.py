from ctypes import CDLL, POINTER, Structure, byref, c_int32, c_int64, c_void_p
from enum import Enum
from typing import List

import torch


class Tensor(Structure):
    _fields_ = [
        ("data", c_void_p),
        ("ndim", c_int32),
        ("shape", POINTER(c_int64)),
        ("strides", POINTER(c_int64)),
        ("dtype", c_int32),
    ]

    class Dtype(Enum):
        INT64 = 0
        FLOAT64 = 1
        INT32 = 2
        FLOAT32 = 3
        FLOAT16 = 4
        BFLOAT16 = 5
        FLOAT8_E4M3FN = 6
        FLOAT8_E4M3FNUZ = 7
        FLOAT8_E5M2 = 8
        FLOAT8_E5M2FNUZ = 9

        @classmethod
        def from_torch_dtype(cls, dtype: torch.dtype):
            return {
                torch.int64: cls.INT64,
                torch.float64: cls.FLOAT64,
                torch.int32: cls.INT32,
                torch.float32: cls.FLOAT32,
                torch.float16: cls.FLOAT16,
                torch.bfloat16: cls.BFLOAT16,
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


class Library(object):

    type_mapping = {
        "t": POINTER(Tensor),
        "i32": c_int32,
        "i64": c_int64,
    }

    def __init__(self, lib_path: str, func_names: List[str], func_params: List[str]):
        self.lib_path = lib_path
        self.func_names = func_names
        self.func_params = func_params

        self._load()

    def _load(self):
        self.lib = CDLL(self.lib_path)
        for func_name, func_param in zip(self.func_names, self.func_params):
            func = getattr(self.lib, func_name)
            # arg 0 of called function is always the cuda stream
            func.argtypes = [c_void_p] + [
                self.type_mapping[p]
                for p in filter(
                    lambda x: len(x.strip()) > 0, func_param.strip().split("_")
                )
            ]
            func.restype = None

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

            self.__setattr__(func_name, _func)
