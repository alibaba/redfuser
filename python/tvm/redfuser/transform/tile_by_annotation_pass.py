"""
TileByAnnotation Pass - 根据循环的 name 注解和 tile_map 进行 tiling

该 pass 会根据循环的 annotations["name"] 查找对应的 tile size，
然后对循环进行 split，将外层循环保留注解，内层循环不保留注解。
"""

import tvm
from tvm import tir
from tvm.tir import PrimFunc
from tvm.ir import IRModule
from typing import Dict, List, Tuple

from .common_analysis_v2 import normalize_prim_func, BlockInfo, IterInfo


def _tile_loops(
    sch: tir.Schedule,
    loops: List[IterInfo],
    decisions: List[int],
) -> Tuple[List[tir.schedule.LoopRV], List[tir.schedule.LoopRV]]:
    """
    对循环列表进行 tiling
    
    Args:
        sch: TIR Schedule
        loops: 循环信息列表
        decisions: 每个循环的 tile size
        
    Returns:
        (outer_loops, inner_loops) 元组
    """
    assert len(loops) == len(decisions)
    new_outer, new_inner = [], []
    
    for loop_info, factor in zip(loops, decisions):
        extent = loop_info.dom
        if extent < factor:
            extents = [1, extent]
        else:
            assert extent % factor == 0, f"{extent} % {factor} != 0"
            outer, inner = extent // factor, factor
            extents = sch.sample_partitioned_tile(
                loop_info.loop_rv[0], 2, 1, 16, decision=[outer, inner]
            )
        i0, i1 = sch.split(loop_info.loop_rv[0], extents)
        new_outer.append(i0)
        new_inner.append(i1)
    
    sch.reorder(*new_outer, *new_inner)
    return new_outer, new_inner


def _tile_block(
    sch: tir.Schedule, 
    block_info: BlockInfo, 
    tile_map: Dict[str, int]
) -> None:
    """
    对单个 block 进行 tiling
    
    Args:
        sch: TIR Schedule
        block_info: block 信息
        tile_map: name -> tile_size 映射
    """
    # 找出需要 tile 的循环（只 tile 在 tile_map 中有对应 name 的循环）
    assert(all(len(iter_info.loop_rv) == 1 for iter_info in block_info.iters))

    tiles = []
    for iter_info in block_info.iters:
        name = iter_info.annotations[0].get("name")
        if name and name in tile_map:
            tiles.append(tile_map[name])
    
    if not tiles:
        return
    
    # 保存原始注解
    original_annotations = [
        iter_info.annotations[0]
        for iter_info in block_info.iters
    ]
    
    # 移除所有循环的注解（split 前需要移除）
    for iter_info in block_info.iters:
        for key in list(iter_info.annotations[0].keys()):
            sch.unannotate(iter_info.loop_rv[0], key)
    
    # 计算 batch loops（不需要 tile 的循环）和需要 tile 的循环的索引
    innermost_n = len(tiles)
    all_iters = block_info.iters
    batch_iters = all_iters[:-innermost_n]
    tile_iters = all_iters[-innermost_n:]
    
    # 保存 batch loop RVs (这些循环不会被 split，所以可以直接使用)
    batch_loop_rvs = [iter_info.loop_rv[0] for iter_info in batch_iters]
    
    # 执行 tiling
    outer_loops, inner_loops = _tile_loops(sch, tile_iters, tiles)
    
    # 重新注解：batch loops + outer loops 保留原始注解
    all_new_loops = batch_loop_rvs + outer_loops
    for i, (ann, new_loop) in enumerate(zip(original_annotations, all_new_loops)):
        for key, value in ann.items():
            sch.annotate(new_loop, key, value)


def _tile_by_annotation_in_func(mod: IRModule, func_name: str, tile_map: Dict[str, int]) -> IRModule:
    """对单个函数执行 tiling"""
    sch = tir.Schedule(mod)
    sch.work_on(func_name)
    
    block_infos = normalize_prim_func(sch)
    if block_infos is None:
        return mod
    
    for block_info in block_infos:
        _tile_block(sch, block_info, tile_map)
    
    return sch.mod


def TileByAnnotation(tile_map: Dict[str, int]):
    """
    创建 TileByAnnotation Pass
    
    Args:
        tile_map: 循环名称到 tile size 的映射，如 {"q_len": 64, "kv_len": 64}
        
    Returns:
        TVM module pass
        
    用法:
        pass_func = tvm.transform.Sequential([
            TileByAnnotation({"q_len": 64, "kv_len": 64})
        ])
        mod = pass_func(mod)
    """
    @tvm.transform.module_pass(opt_level=0, name="TileByAnnotation")
    def _pass(mod: IRModule, ctx: tvm.transform.PassContext) -> IRModule:
        for gvar, func in mod.functions.items():
            if isinstance(func, PrimFunc):
                mod = _tile_by_annotation_in_func(mod, gvar.name_hint, tile_map)
        return mod
    
    return _pass
