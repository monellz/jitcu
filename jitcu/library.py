from ctypes import Structure, POINTER, c_void_p, c_int32, c_int64, CDLL, byref
from typing import List
import torch

class Tensor(Structure):
  _fields_ = [
    ('data', c_void_p),
    ('ndim', c_int32),
    ('shape', POINTER(c_int64)),
    ('strides', POINTER(c_int64)),
  ]

  @classmethod
  def from_torch_tensor(cls, src):
    assert src.storage_offset() == 0
    data = c_void_p(src.data_ptr())
    ndim = src.dim()
    shape = (c_int64 * ndim)(*src.shape)
    strides = (c_int64 * ndim)(*src.stride())
    return cls(
      data=data,
      ndim=ndim,
      shape=shape,
      strides=strides,
    )


class Library(object):

  type_mapping = {
    't': POINTER(Tensor),
    'i32': c_int32,
    'i64': c_int64,
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
      func.argtypes = [self.type_mapping[p] for p in func_param.split('_')]
      func.restype = None

      def _func(*args):
        args = [
          byref(Tensor.from_torch_tensor(arg)) if isinstance(arg, torch.Tensor) else arg
          for arg in args
        ]
        return func(*args)
      self.__setattr__(func_name, _func)
