# gemm_base.py

import torch
import triton
import triton.language as tl

from abc import ABC, abstractmethod

class GEMMKernelBase(ABC):
    """
    抽象基类，声明三个接口：
      - prepare(m, k, n, should_gen_data=True) -> params(dict)
      - run(params, repeats=1) -> elapsed_seconds (float)
      - verify(params, rtol=1e-5, atol=1e-5) -> bool
    子类需要实现这些方法。
    """

    DTYPE_CONFIG = {
        "i8":   (torch.int8,   tl.int8),
        "i16":  (torch.int16,  tl.int16),
        "i32":  (torch.int32,  tl.int32),
        "fp16": (torch.float16, tl.float16),
        "bf16": (torch.bfloat16, tl.bfloat16),
        "fp32": (torch.float32, tl.float32),
    }

    BITWIDTH = {
        "i8": 8,
        "i16": 16,
        "i32": 32,
        "fp16": 16,
        "bf16": 16,
        "fp32": 32,
    }

    @abstractmethod
    def prepare(self, m: int, k: int, n: int, should_gen_data: bool = True):
        """
        准备数据。返回一个参数列表 (即run和verify中的params)。
        """
        raise NotImplementedError

    @abstractmethod
    def run(self, params: dict, repeats: int = 1) -> float:
        """
        执行 kernel ，重复 repeats 次。
        返回平均耗时（秒）。
        """
        raise NotImplementedError

    @abstractmethod
    def verify(self, params: dict) -> bool:
        """
        验证结果正确性。
        返回 True/False。
        """
        raise NotImplementedError

    @abstractmethod
    def get_name(self) -> int:
        """
        返回一个可以唯一标识该kernel的名称。通常包含kernel名称和数据类型标识。
        """
        return self.__class__.__name__

    @abstractmethod
    def expected_cycles(self, m: int, k: int, n: int) -> int:
        """
        返回理论上该 kernel 执行 m x k x n 大小的 GEMM 所需的 CPU 周期数。
        """
        raise NotImplementedError
