"""
TileByAnnotation Pass - 根据循环的 name 注解和 tile_map 进行 tiling

该 pass 会根据循环的 annotations["name"] 查找对应的 tile size，
然后对循环进行 split，将外层循环保留注解，内层循环不保留注解。
"""

import tvm
from tvm import tir
from tvm.tir import PrimFunc
from tvm.ir import IRModule
from typing import Dict, List

from .common_analysis_v2 import normalize_prim_func, BlockInfo


def _tile_block(
    sch: tir.Schedule,
    block_info: BlockInfo,
    tile_map: Dict[str, int]
) -> None:
    """
    对单个 block 进行 tiling

    直接遍历 block 上方的所有 loop，根据 loop 的 annotations["name"] 匹配 tile_map，
    对匹配的 loop 执行 split。这样可以正确处理一个 iter var 绑定多个 loop 的情况
    （如 v_kv_len = T.axis.spatial(512, split * 128 + kv_len)）。

    Args:
        sch: TIR Schedule
        block_info: block 信息
        tile_map: name -> tile_size 映射
    """
    block_rv = block_info.block_rv
    loop_rvs = list(sch.get_loops(block_rv))

    # Step 1: 分析所有 loop，收集注解并判断是否需要 tile
    loop_infos = []  # [(loop_rv, annotations_dict, tile_factor_or_None)]
    for loop_rv in loop_rvs:
        loop = sch.get(loop_rv)
        annotations = dict(loop.annotations)
        name = str(annotations.get("name", ""))
        tile_factor = tile_map.get(name) if name else None
        loop_infos.append((loop_rv, annotations, tile_factor))

    # 如果没有需要 tile 的 loop，直接返回
    if not any(info[2] is not None for info in loop_infos):
        return

    # Step 2: 移除所有 loop 的注解（split 前需要移除）
    for loop_rv, annotations, _ in loop_infos:
        for key in list(annotations.keys()):
            sch.unannotate(loop_rv, key)

    # Step 3: 对需要 tile 的 loop 执行 split，不需要 tile 的保持原样
    # 保持原始顺序：记录每个位置是 (loop_rv, annotations) 还是被 split 后的 (outer_rv, annotations)
    ordered_outer_loops = []  # [(loop_rv, annotations, inner_var)] 按原始顺序，包含非 tile 和 outer
    inner_loops = []          # [loop_rv] 所有 inner loops

    for loop_rv, annotations, tile_factor in loop_infos:
        if tile_factor is None:
            ordered_outer_loops.append((loop_rv, annotations, None))
        else:
            loop = sch.get(loop_rv)
            extent = int(loop.extent)
            if extent <= tile_factor:
                extents = [1, extent]
            else:
                assert extent % tile_factor == 0, f"{extent} % {tile_factor} != 0"
                outer_size, inner_size = extent // tile_factor, tile_factor
                extents = [outer_size, inner_size]
            i0, i1 = sch.split(loop_rv, extents)
            ordered_outer_loops.append((i0, annotations, sch.get(i1).loop_var.name))
            inner_loops.append(i1)

    # Step 4: Reorder - 保持非 tile 和 outer loops 的原始顺序，inner loops 放到最内层
    all_ordered = [lv for lv, _, _ in ordered_outer_loops] + inner_loops
    sch.reorder(*all_ordered)

    # Step 5: 重新注解 - 恢复原始注解
    for loop_rv, annotations, inner_var in ordered_outer_loops:
        if inner_var is not None:
            sch.annotate(loop_rv, "tiled", inner_var)
        for key, value in annotations.items():
            sch.annotate(loop_rv, key, value)


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
