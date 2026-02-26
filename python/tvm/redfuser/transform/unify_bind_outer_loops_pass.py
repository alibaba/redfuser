"""
UnifyBindOuterLoops Pass - 统一所有 block 的外层 bind 循环结构

功能:
1. 收集所有带有 bind 注解的循环，按 bind 值降序排序
2. 对于缺少某些 bind 循环的 block，插入与原始循环完全相同的 For 循环
3. 在 Block 中添加对应的 iter_var 映射
"""

import tvm
from tvm import tir
from tvm.tir import PrimFunc, For, Block, BlockRealize, IterVar
from tvm.ir import IRModule, Range
from typing import List, Tuple, Set


def _collect_bind_loop(sch: tir.Schedule) -> List[Tuple[str, tir.For]]:
    """收集所有带有 bind 注解的循环，按 bind 值降序排序并去重（同一 bind 只保留第一个）"""
    block_rvs = sch.get_child_blocks(sch.get_block("root"))
    seen_binds: Set[str] = set()
    bind_loop_info: List[Tuple[str, tir.For]] = []

    for block_rv in block_rvs:
        for loop_rv in sch.get_loops(block_rv):
            loop = sch.get(loop_rv)
            if "bind" in loop.annotations:
                bind_val = str(loop.annotations["bind"])
                if bind_val not in seen_binds:
                    seen_binds.add(bind_val)
                    bind_loop_info.append((bind_val, loop))

    return sorted(bind_loop_info, key=lambda x: x[0], reverse=True)


def _insert_missing_bind_loops(
    sch: tir.Schedule, sorted_bind_loops: List[Tuple[str, tir.For]]
) -> List[Tuple[tir.Var, tir.For]]:
    """
    为每个 block 插入缺失的 bind 循环（unit loop）
    返回需要替换的 (unit_loop_var, target_loop) 列表
    """
    block_rvs = sch.get_child_blocks(sch.get_block("root"))
    replacements: List[Tuple[tir.Var, tir.For]] = []

    for block_rv in block_rvs:
        existing_binds: Set[str] = set()
        for lrv in sch.get_loops(block_rv):
            loop = sch.get(lrv)
            if "bind" in loop.annotations:
                existing_binds.add(str(loop.annotations["bind"]))

        missing = [bl for bl in sorted_bind_loops if bl[0] not in existing_binds]
        if not missing:
            continue

        for miss_bind, miss_loop in missing:
            loop_rvs = list(sch.get_loops(block_rv))

            insert_target = None
            for lrv in loop_rvs:
                loop = sch.get(lrv)
                bind_val = loop.annotations.get("bind")
                if bind_val is not None:
                    if str(bind_val) < miss_bind:
                        insert_target = lrv
                        break
                else:
                    insert_target = lrv
                    break

            new_loop_rv = sch.add_unit_loop(insert_target if insert_target else block_rv)
            unit_loop = sch.get(new_loop_rv)
            replacements.append((unit_loop.loop_var, miss_loop))

    return replacements


def _add_iter_var_to_block(
    block_realize: BlockRealize,
    new_loop_var: tir.Var,
    target_loop: tir.For,
    sorted_bind_loops: List[Tuple[str, tir.For]],
) -> BlockRealize:
    """在 BlockRealize 中按正确位置添加新的 iter_var 映射"""
    block = block_realize.block
    target_bind = str(target_loop.annotations.get("bind", ""))

    new_iter_var = IterVar(
        dom=Range(target_loop.min, target_loop.extent),
        var=tir.Var("v_" + new_loop_var.name, new_loop_var.dtype),
        iter_type=IterVar.DataPar,
    )

    iter_vars = list(block.iter_vars)
    iter_values = list(block_realize.iter_values)

    insert_pos = len(iter_vars)
    for i, (bind_val, _) in enumerate(sorted_bind_loops):
        if bind_val == target_bind:
            insert_pos = i
            break

    new_iter_vars = iter_vars[:insert_pos] + [new_iter_var] + iter_vars[insert_pos:]
    new_iter_values = iter_values[:insert_pos] + [new_loop_var] + iter_values[insert_pos:]

    new_block = Block(
        new_iter_vars,
        block.reads,
        block.writes,
        block.name_hint,
        block.body,
        block.init,
        block.alloc_buffers,
        block.match_buffers,
        block.annotations,
    )

    return BlockRealize(new_iter_values, block_realize.predicate, new_block)


def _replace_unit_loops(
    func: PrimFunc,
    replacements: List[Tuple[tir.Var, tir.For]],
    sorted_bind_loops: List[Tuple[str, tir.For]],
) -> PrimFunc:
    """替换 unit loop 为目标循环，并在内部 Block 添加 iter_var 映射"""
    if not replacements:
        return func

    new_body = func.body
    for unit_var, target_loop in replacements:
        new_loop_var = tir.Var(target_loop.loop_var.name, target_loop.loop_var.dtype)

        current_unit_var = unit_var
        current_new_var = new_loop_var
        current_target = target_loop
        current_sorted_binds = sorted_bind_loops

        def replace_for(op):
            if isinstance(op, For) and op.loop_var.same_as(current_unit_var):
                new_for_body = op.body

                def add_iter_var_to_blocks(stmt):
                    if isinstance(stmt, BlockRealize):
                        return _add_iter_var_to_block(
                            stmt, current_new_var, current_target, current_sorted_binds
                        )
                    return None

                new_for_body = tir.stmt_functor.ir_transform(
                    new_for_body, None, add_iter_var_to_blocks, ["tir.BlockRealize"]
                )

                return For(
                    current_new_var,
                    current_target.min,
                    current_target.extent,
                    current_target.kind,
                    new_for_body,
                    current_target.thread_binding,
                    current_target.annotations,
                )
            return None

        new_body = tir.stmt_functor.ir_transform(new_body, None, replace_for, ["tir.For"])

    return func.with_body(new_body)


def _unify_bind_outer_loops_in_func(mod: IRModule, func_name: str) -> IRModule:
    """对单个函数执行 bind 循环统一"""
    sch = tir.Schedule(mod, error_render_level="fast", enable_check=False)
    sch.work_on(func_name)

    sorted_bind_loops = _collect_bind_loop(sch)
    if not sorted_bind_loops:
        return mod

    replacements = _insert_missing_bind_loops(sch, sorted_bind_loops)
    new_func = _replace_unit_loops(sch.mod[func_name], replacements, sorted_bind_loops)

    new_functions = {
        gvar: (new_func if gvar.name_hint == func_name else f)
        for gvar, f in sch.mod.functions.items()
    }
    return IRModule(new_functions)


@tvm.transform.module_pass(opt_level=0, name="UnifyBindOuterLoops")
def UnifyBindOuterLoops(mod: IRModule, ctx: tvm.transform.PassContext) -> IRModule:
    """TVM Pass: 统一所有 block 的外层 bind 循环结构"""
    for gvar, func in mod.functions.items():
        if isinstance(func, PrimFunc):
            mod = _unify_bind_outer_loops_in_func(mod, gvar.name_hint)
    return mod
