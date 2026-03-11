from typing import Optional

import tvm
from tvm import dlight as dl
from tvm.tir import PrimFunc
from tvm.ir import IRModule

from .common_analysis_v2 import normalize_prim_func
from .tir_reduction_analyzer import analyze_cascaded_group
from .decompose import ReductionProcessor
from .tir_online_fusion_generator import transform_reductions
from .tir_online_fusion_with_split_generator import transform_reductions_with_split


def _generate_online_expr(mod: IRModule, func_name: str, num_split: Optional[int] = None) -> IRModule:
    # do schedule here
    sch = tvm.tir.Schedule(mod)
    sch.work_on(func_name)

    # pre-process
    all_blocks = normalize_prim_func(sch)
    _ = dl.try_inline(sch, all_blocks)

    # find all cascaede reduction groups
    all_blocks = normalize_prim_func(sch)
    reduction_blocks = [(idx, block) for idx, block in enumerate(all_blocks) if block.is_reduction()]

    cascaded_groups = []
    i = 0

    def _get_reduction_extent(block):
        for iter_info in block.iters:
            if iter_info.kind == "R":
                return iter_info.dom
        return None

    while i < len(reduction_blocks):
        group_indices = [reduction_blocks[i][0]]
        group_blocks = [reduction_blocks[i][1]]
        extent = _get_reduction_extent(group_blocks[0])

        j = i + 1
        while j < len(reduction_blocks):
            curr_idx, curr_block = reduction_blocks[j]
            prev_idx = reduction_blocks[j - 1][0]

            if curr_idx == prev_idx + 1 and _get_reduction_extent(curr_block) == extent:
                group_indices.append(curr_idx)
                group_blocks.append(curr_block)
                j += 1
            else:
                break

        if len(group_blocks) >= 2:
            cascaded_groups.append(group_blocks)

        i = j if j > i + 1 else i + 1

    # FIXME(liyangcheng.lyc)
    # 目前代码的设计和实现中蕴含的一些假设:
    # 1. 当实现方式是non-split时,允许子图中存在多个级联规约的结构,会遍历每一个级联规约结构对其进行变换
    # 2. 当实现方式是split时,则要求子图中有且仅有一个结构:即唯一的一个级联规约结构,原因是split会产生两个kernel,没有考虑这个子图如果存在其他op的情况
    # 3. 以上两种情况之外的子图,目前应该无法处理
    if num_split is not None:
        assert len(cascaded_groups) == 1

        reduction_infos = analyze_cascaded_group(cascaded_groups[0], sch)
            
        reduction_configs = reduction_infos.reduction_configs
        x_map = reduction_infos.x_map
        y_map = reduction_infos.y_map
        c_map = reduction_infos.c_map

        reduction_processor = ReductionProcessor(reduction_configs, x_map, y_map, c_map)
        exprs_list, reduce_funcs_list = reduction_processor.process_reductions()

        # 生成带Split的Online算法的TIR Block,应当返回一个包含两个函数的新Module(?)(split + combine)
        transform_reductions_with_split(reduction_infos, reduce_funcs_list, num_split)

        return sch.mod

    else: # num_split is None
        # for every possible cascaded group, get their info
        for idx, cascaded_group in enumerate(cascaded_groups):
            reduction_infos = analyze_cascaded_group(cascaded_group, sch)
                
            reduction_configs = reduction_infos.reduction_configs
            x_map = reduction_infos.x_map
            y_map = reduction_infos.y_map
            c_map = reduction_infos.c_map

            reduction_processor = ReductionProcessor(reduction_configs, x_map, y_map, c_map)
            exprs_list, reduce_funcs_list = reduction_processor.process_reductions()

            # 生成Online算法的TIR Block,结束时sch已经被改变了
            transform_reductions(reduction_infos, reduce_funcs_list)

        return sch.mod


def GenerateOnlineExpr(num_split: Optional[int] = None):
    @tvm.transform.module_pass(opt_level=0, name="GenerateOnlineExpr")
    def _pass(mod: IRModule, ctx: tvm.transform.PassContext) -> IRModule:
        for gvar, func in mod.functions.items():
            if isinstance(func, PrimFunc):
                mod = _generate_online_expr(mod, gvar.name_hint, num_split)
        return mod

    return _pass
