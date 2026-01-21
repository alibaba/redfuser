"""
DecomposeReduction Pass - 将 reduction block 分解为 init 和 update 两部分
"""

import tvm
from tvm import tir
from tvm.tir import PrimFunc
from tvm.ir import IRModule

from .common_analysis_v2 import normalize_prim_func, BlockInfo


def _find_insert_loop(sch: tir.Schedule, block_info: BlockInfo) -> tir.schedule.LoopRV:
    """
    找到 init block 应该插入的循环位置。
    
    规则：
    1. 如果 reduction axis 有 "name" 注解，在该循环之前插入
    2. 否则在第一个没有 "name" 注解的循环之前插入
    """
    block_rv = sch.get_block(block_info.name)
    loops = sch.get_loops(block_rv)
    
    # 找第一个 reduction axis 的 name 注解
    reduction_loop_name = None
    for iter_info in block_info.iters:
        if iter_info.kind == "R":
            for ann in iter_info.annotations:
                if ann.get("name"):
                    reduction_loop_name = ann.get("name")
                    break
            break
    
    # 查找目标循环
    for loop_rv in loops:
        loop = sch.get(loop_rv)
        if reduction_loop_name:
            # 找名称匹配的循环
            if loop.annotations.get("name") == reduction_loop_name:
                return loop_rv
        else:
            # 找第一个没有 name 注解的循环
            if not loop.annotations.get("name"):
                return loop_rv
    
    return loops[0]


def _decompose_reduction_in_func(mod: IRModule, func_name: str) -> IRModule:
    """对单个函数执行 reduction 分解"""
    sch = tir.Schedule(mod, error_render_level="fast", enable_check=False)
    sch.work_on(func_name)
    
    block_infos = normalize_prim_func(sch)
    if not block_infos:
        return mod
    
    # 收集 reduction blocks 的名称
    reduction_names = [b.name for b in block_infos if b.is_reduction()]
    
    # 逐个处理
    for block_info in block_infos:
        if block_info.name not in reduction_names:
            continue
        try:
            insert_loop = _find_insert_loop(sch, block_info)
            block_rv = sch.get_block(block_info.name)
            sch.decompose_reduction(block_rv, insert_loop)
        except Exception:
            pass
    
    return sch.mod


@tvm.transform.module_pass(opt_level=0, name="DecomposeReduction")
def DecomposeReduction(mod: IRModule, ctx: tvm.transform.PassContext) -> IRModule:
    """TVM Pass: 将 reduction block 分解为 init 和 update 两部分"""
    for gvar, func in mod.functions.items():
        if isinstance(func, PrimFunc):
            mod = _decompose_reduction_in_func(mod, gvar.name_hint)
    return mod
