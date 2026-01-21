"""
BlockizeInnerLoops Pass - 对没有注解的内层循环执行 blockize

该 pass 会遍历所有 block，找到第一个没有注解的循环并执行 blockize。
"""

import tvm
from tvm import tir
from tvm.tir import PrimFunc
from tvm.ir import IRModule

def _blockize_inner_loops_in_func(mod: IRModule, func_name: str) -> IRModule:
    """对单个函数执行 blockize"""
    sch = tir.Schedule(mod)
    sch.work_on(func_name)
    
    root_block_rv = sch.get_block("root")
    block_rvs = sch.get_child_blocks(root_block_rv)
    
    for block_rv in block_rvs:
        loop_rvs = list(sch.get_loops(block_rv))
        for loop_rv in loop_rvs:
            loop = sch.get(loop_rv)
            # 找到第一个没有注解的循环
            if len(loop.annotations) == 0:
                sch.blockize(loop_rv)
                break
    
    return sch.mod


@tvm.transform.module_pass(opt_level=0, name="BlockizeInnerLoops")
def BlockizeInnerLoops(mod: IRModule, ctx: tvm.transform.PassContext) -> IRModule:
    """
    TVM Pass: 对没有注解的内层循环执行 blockize
    
    用法:
        pass_func = tvm.transform.Sequential([BlockizeInnerLoops])
        mod = pass_func(mod)
    """
    for gvar, func in mod.functions.items():
        if isinstance(func, PrimFunc):
            mod = _blockize_inner_loops_in_func(mod, gvar.name_hint)
    return mod
