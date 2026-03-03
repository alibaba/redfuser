"""
Merge Fused Loop Pass - 合并带有 {"tag": "fused"} 注解的循环

功能:
1. 合并具有相同 bind 注解的循环
2. 合并带有 {"tag": "fused"} 注解的顶层循环
3. 合并融合循环内部的子循环
4. 合并融合循环内部的 blocks
"""

import tvm
from tvm import tir
from tvm.tir import PrimFunc, For, Block, BlockRealize, SeqStmt, Stmt
from tvm.ir import IRModule
from typing import List, Optional, Dict


def _merge_loops_at_level(sch: tir.Schedule, loops: List[tir.schedule.LoopRV]) -> tir.schedule.LoopRV:
    """合并循环列表，保留第一个循环的注解"""
    if len(loops) < 2:
        return loops[0]
    
    annotations = dict(sch.get(loops[0]).annotations)
    for loop in loops:
        for key in list(sch.get(loop).annotations.keys()):
            sch.unannotate(loop, key)
    
    merged = sch.merge(*loops)
    for key, value in annotations.items():
        sch.annotate(merged, key, value)
    return merged


def _merge_loops_by_bind(sch: tir.Schedule, func_name: str) -> None:
    """合并具有相同 bind 注解的循环"""
    sch.work_on(func_name)
    block_rvs = sch.get_child_blocks(sch.get_block("root"))
    blocks_loop_rvs = [list(sch.get_loops(b)) for b in block_rvs]
    
    if not blocks_loop_rvs:
        return
    
    for level in range(max(len(loops) for loops in blocks_loop_rvs)):
        valid = [loops for loops in blocks_loop_rvs if level < len(loops)]
        if not valid:
            continue
        
        binding = sch.get(valid[0][level]).annotations.get("bind", "empty_bind")
        if all(sch.get(loops[level]).annotations.get("bind") == binding for loops in valid):
            loops_to_merge = [loops[level] for loops in valid]
            if len(loops_to_merge) >= 2:
                _merge_loops_at_level(sch, loops_to_merge)


def _merge_fused_top_loops(sch: tir.Schedule, func_name: str) -> Optional[tir.schedule.LoopRV]:
    """合并带有 tag=fused 注解的顶层循环"""
    sch.work_on(func_name)
    block_rvs = sch.get_child_blocks(sch.get_block("root"))
    blocks_loop_rvs = [list(sch.get_loops(b)) for b in block_rvs]
    
    if not blocks_loop_rvs:
        return None
    
    for level in range(max(len(loops) for loops in blocks_loop_rvs)):
        valid = [loops for loops in blocks_loop_rvs if level < len(loops)]
        if not valid:
            continue
        
        # 找带有 fused tag 的循环
        fused = [loops for loops in valid 
                 if sch.get(loops[level]).annotations.get("tag") == "fused"]
        if not fused:
            continue
        
        loops_to_merge = [loops[level] for loops in fused]
        
        # 临时移除父循环注解
        parent_backup = []
        if level > 0:
            for loops in fused:
                parent = loops[level - 1]
                ann = dict(sch.get(parent).annotations)
                parent_backup.append((parent, ann))
                for key in ann:
                    sch.unannotate(parent, key)
        
        merged = _merge_loops_at_level(sch, loops_to_merge)
        
        # 恢复父循环注解
        for parent, ann in parent_backup:
            for key, value in ann.items():
                sch.annotate(parent, key, value)
        
        return merged
    return None


def _collect_reduction_blocks_info(
    sch: tir.Schedule, fused_loop_rv: tir.schedule.LoopRV
) -> tuple:
    """
    步骤1: 收集 fused loop 内部名字以 "reduction" 开头的 block 的 inner_loops
    
    返回: (sorted_inner_loops, block_info)
        - sorted_inner_loops: 按 loop_var.name 去重后的目标循环列表
        - block_info: [(block_name, inner_loops_as_For), ...]
    """
    child_blocks = sch.get_child_blocks(fused_loop_rv)
    
    seen_loop_vars: set = set()
    sorted_inner_loops: List[tir.For] = []
    block_info: List[tuple] = []
    
    for block_rv in child_blocks:
        block_name = sch.get(block_rv).name_hint
        if not block_name.startswith("reduction"):
            continue
        
        block_loops = list(sch.get_loops(block_rv))
        
        for i, loop_rv in enumerate(block_loops):
            if sch.get_sref(loop_rv).same_as(sch.get_sref(fused_loop_rv)):
                inner_loop_rvs = block_loops[i + 1:]
                inner_loops_as_for = []
                
                for lrv in inner_loop_rvs:
                    loop = sch.get(lrv)
                    if isinstance(loop, tir.For):
                        inner_loops_as_for.append(loop)
                        loop_var_name = loop.loop_var.name
                        if loop_var_name not in seen_loop_vars:
                            seen_loop_vars.add(loop_var_name)
                            sorted_inner_loops.append(loop)
                
                block_info.append((block_name, inner_loops_as_for))
                break
    
    return sorted_inner_loops, block_info


def _get_missing_loops_per_block(
    block_info: List[tuple],
    sorted_inner_loops: List[tir.For]
) -> Dict[str, List[tir.For]]:
    """
    计算每个 block 缺失的循环
    
    参数:
        block_info: [(block_name, inner_loops_as_For), ...]
        sorted_inner_loops: 去重后的目标循环列表
    
    返回: {block_name: [missing_loop_1, missing_loop_2, ...], ...}
    """
    missing_loops_map: Dict[str, List[tir.For]] = {}
    
    for block_name, inner_loops in block_info:
        existing_loop_vars: set = set()
        for loop in inner_loops:
            existing_loop_vars.add(loop.loop_var.name)
        
        missing_loops = [
            loop for loop in sorted_inner_loops 
            if loop.loop_var.name not in existing_loop_vars
        ]
        
        if missing_loops:
            missing_loops_map[block_name] = missing_loops
    
    return missing_loops_map


def _align_inner_loops_in_func(
    func: PrimFunc,
    missing_loops_map: Dict[str, List[tir.For]]
) -> PrimFunc:
    """
    步骤2: 在 IR 层面直接为每个 reduction block 插入缺失的循环
    
    在 BlockRealize 外层插入缺失的 For 循环，不添加 block 内部的 axis 映射
    """
    if not missing_loops_map:
        return func
    
    def insert_loops(stmt):
        if isinstance(stmt, BlockRealize):
            block_name = stmt.block.name_hint
            if block_name in missing_loops_map:
                missing_loops = missing_loops_map[block_name]
                result = stmt
                for target_loop in reversed(missing_loops):
                    result = For(
                        target_loop.loop_var,
                        target_loop.min,
                        target_loop.extent,
                        target_loop.kind,
                        result,
                        target_loop.thread_binding,
                        target_loop.annotations,
                    )
                return result
        return None
    
    new_body = tir.stmt_functor.ir_transform(
        func.body, None, insert_loops, ["tir.BlockRealize"]
    )
    
    return func.with_body(new_body)


def _topological_sort_block_names(
    sch: tir.Schedule,
    fused_loop_rv: tir.schedule.LoopRV,
    block_names: set
) -> List[str]:
    """
    步骤3: 按 producer -> consumer 顺序对 block names 进行拓扑排序
    """
    if len(block_names) <= 1:
        return list(block_names)
    
    child_blocks = sch.get_child_blocks(fused_loop_rv)
    name_to_rv = {}
    for block_rv in child_blocks:
        name = sch.get(block_rv).name_hint
        if name in block_names:
            name_to_rv[name] = block_rv
    
    in_degree: Dict[str, int] = {name: 0 for name in block_names}
    successors: Dict[str, List[str]] = {name: [] for name in block_names}
    
    for name in block_names:
        block_rv = name_to_rv.get(name)
        if block_rv is None:
            continue
        consumers = sch.get_consumers(block_rv)
        for consumer in consumers:
            consumer_name = sch.get(consumer).name_hint
            if consumer_name in block_names and consumer_name != name:
                successors[name].append(consumer_name)
                in_degree[consumer_name] += 1
    
    queue = [name for name in block_names if in_degree[name] == 0]
    sorted_names = []
    
    while queue:
        queue.sort()
        current = queue.pop(0)
        sorted_names.append(current)
        
        for succ in successors[current]:
            in_degree[succ] -= 1
            if in_degree[succ] == 0:
                queue.append(succ)
    
    if len(sorted_names) != len(block_names):
        return list(block_names)
    
    return sorted_names


def _merge_reduction_blocks_in_func(
    func: PrimFunc,
    sorted_block_names: List[str],
    sorted_inner_loops: List[tir.For]
) -> PrimFunc:
    """
    按拓扑顺序将所有 reduction block 直接融合为一个 block, 
    并根据 sorted_inner_loops 将融合后的 block 放到对应的 for 循环下。
    """
    if len(sorted_block_names) < 2 or not sorted_inner_loops:
        return func
    
    target_block_names = set(sorted_block_names)
    
    canonical_loop_var_map: Dict[str, tir.Var] = {}
    for loop in sorted_inner_loops:
        canonical_loop_var_map[loop.loop_var.name] = loop.loop_var
    
    def merge_blocks_under_fused_loop(stmt):
        if not (isinstance(stmt, For) and stmt.annotations.get("tag") == "fused"):
            return None
        
        collected_block_realizes: Dict[str, BlockRealize] = {}
        block_loop_vars: Dict[str, List[tir.Var]] = {}
        
        def collect_and_rebuild(s, current_loop_vars: List[tir.Var] = None) -> tuple:
            """收集 reduction blocks 并重建剩余语句结构，返回 (new_stmt_or_list, found_or_insert_pos)"""
            if current_loop_vars is None:
                current_loop_vars = []
            
            if isinstance(s, BlockRealize):
                block_name = s.block.name_hint
                if block_name in target_block_names:
                    collected_block_realizes[block_name] = s
                    block_loop_vars[block_name] = list(current_loop_vars)
                    return (None, True)
                return (s, False)
            elif isinstance(s, For):
                new_loop_vars = current_loop_vars + [s.loop_var]
                new_body, found = collect_and_rebuild(s.body, new_loop_vars)
                if found and new_body is None:
                    return (None, True)
                elif new_body is not s.body:
                    return (For(s.loop_var, s.min, s.extent, s.kind,
                               new_body, s.thread_binding, s.annotations), found)
                return (s, found)
            elif isinstance(s, SeqStmt):
                new_seq = []
                any_found = False
                first_reduction_pos = -1
                
                for sub in s.seq:
                    new_sub, found = collect_and_rebuild(sub, current_loop_vars)
                    if found:
                        any_found = True
                        if first_reduction_pos < 0:
                            first_reduction_pos = len(new_seq)
                    if new_sub is not None:
                        new_seq.append(new_sub)
                
                if any_found and first_reduction_pos >= 0:
                    return (new_seq, first_reduction_pos)
                elif len(new_seq) == 0:
                    return (None, any_found)
                elif len(new_seq) == 1:
                    return (new_seq[0], any_found)
                else:
                    return (SeqStmt(new_seq), any_found)
            return (s, False)
        
        result = collect_and_rebuild(stmt.body)
        
        if len(collected_block_realizes) < 2:
            return None
        
        # 按拓扑顺序收集 BlockRealize，并将循环变量替换为 canonical 变量
        sorted_brs = []
        for name in sorted_block_names:
            if name not in collected_block_realizes:
                continue
            br = collected_block_realizes[name]
            loop_vars = block_loop_vars.get(name, [])
            
            var_map: Dict[tir.Var, tir.PrimExpr] = {}
            for lv in loop_vars:
                if lv.name in canonical_loop_var_map:
                    canonical_var = canonical_loop_var_map[lv.name]
                    if not lv.same_as(canonical_var):
                        var_map[lv] = canonical_var
            
            if var_map:
                new_iter_values = [
                    tir.stmt_functor.substitute(iv, var_map) 
                    for iv in br.iter_values
                ]
                br = BlockRealize(new_iter_values, br.predicate, br.block)
            
            sorted_brs.append(br)

        # 直接融合所有 reduction blocks 为一个 block
        merged_body = _merge_block_group(sorted_brs)

        # 用 sorted_inner_loops 包裹融合后的 block
        for target_loop in reversed(sorted_inner_loops):
            merged_body = For(
                target_loop.loop_var,
                target_loop.min,
                target_loop.extent,
                target_loop.kind,
                merged_body,
                target_loop.thread_binding,
                target_loop.annotations,
            )
        
        # 将融合结果插入到第一个 reduction block 的原始位置
        if isinstance(result, tuple) and isinstance(result[0], list):
            new_seq = result[0]
            insert_pos = result[1]
            new_seq.insert(insert_pos, merged_body)
            new_body = SeqStmt(new_seq) if len(new_seq) > 1 else new_seq[0]
        elif result[0] is None:
            new_body = merged_body
        else:
            new_body = SeqStmt([merged_body, result[0]])
        
        return For(
            stmt.loop_var, stmt.min, stmt.extent, stmt.kind,
            new_body, stmt.thread_binding, stmt.annotations
        )
    
    new_body = tir.stmt_functor.ir_transform(
        func.body, None, merge_blocks_under_fused_loop, ["tir.For"]
    )
    
    return func.with_body(new_body)


def _process_reduction_blocks(
    sch: tir.Schedule, fused_loop_rv: tir.schedule.LoopRV
) -> tuple:
    """
    处理 fused loop 内部的 reduction blocks
    
    步骤:
    1. 收集所有 reduction block 的 inner_loops
    2. 计算每个 block 缺失的循环（用于对齐）
    3. 按 producer -> consumer 拓扑排序
    
    返回: (sorted_inner_loops, missing_loops_map, sorted_block_names)
    """
    sorted_inner_loops, block_info = _collect_reduction_blocks_info(sch, fused_loop_rv)
    
    if len(block_info) < 2:
        return sorted_inner_loops, {}, [info[0] for info in block_info]
    
    missing_loops_map = _get_missing_loops_per_block(block_info, sorted_inner_loops)
    
    block_names = set(info[0] for info in block_info)
    sorted_block_names = _topological_sort_block_names(sch, fused_loop_rv, block_names)
    
    return sorted_inner_loops, missing_loops_map, sorted_block_names


def _substitute_buffer_region(
    buf_region: tir.BufferRegion, var_map: Dict[tir.Var, tir.PrimExpr]
) -> tir.BufferRegion:
    """替换 BufferRegion 中的变量"""
    new_region = [
        tvm.ir.Range.from_min_extent(
            tir.stmt_functor.substitute(rng.min, var_map),
            tir.stmt_functor.substitute(rng.extent, var_map)
        ) for rng in buf_region.region
    ]
    return tir.BufferRegion(buf_region.buffer, new_region)


def _merge_block_group(blocks: List[BlockRealize]) -> Stmt:
    """将一组 BlockRealize 融合为单个 BlockRealize"""
    if len(blocks) == 1:
        return blocks[0]
    
    base_idx = 0
    max_iter_vars = len(blocks[0].block.iter_vars)
    for i, br in enumerate(blocks):
        if len(br.block.iter_vars) > max_iter_vars:
            max_iter_vars = len(br.block.iter_vars)
            base_idx = i
    
    base_br = blocks[base_idx]
    base_block = base_br.block
    base_iter_vars = base_block.iter_vars
    base_iter_vars_map = {iv.var.name:iv.var for iv in base_iter_vars}
    base_iter_values = base_br.iter_values
    
    body_stmts, init_stmts = [], []
    all_reads, all_writes = [], []
    all_alloc_buffers = []
    
    for idx, br in enumerate(blocks):
        block = br.block
        
        if idx == base_idx:
            body_stmts.append(block.body)
            if block.init:
                init_stmts.append(block.init)
            all_reads.extend(block.reads)
            all_writes.extend(block.writes)
        else:
            var_map: Dict[tir.Var, tir.PrimExpr] = {}
            for iv_cur in block.iter_vars:
                var_map[iv_cur.var] = base_iter_vars_map[iv_cur.var.name]
            
            body_stmts.append(tir.stmt_functor.substitute(block.body, var_map) if var_map else block.body)
            if block.init:
                init_stmts.append(
                    tir.stmt_functor.substitute(block.init, var_map) if var_map else block.init
                )
            all_reads.extend(
                _substitute_buffer_region(r, var_map) if var_map else r
                for r in block.reads
            )
            all_writes.extend(
                _substitute_buffer_region(w, var_map) if var_map else w
                for w in block.writes
            )
        
        all_alloc_buffers.extend(block.alloc_buffers)
    
    reads_dict = {r.buffer.data: r for r in all_reads}
    writes_dict = {w.buffer.data: w for w in all_writes}
    
    merged_block = Block(
        base_iter_vars,
        list(reads_dict.values()),
        list(writes_dict.values()),
        "reductions_merged",
        SeqStmt(body_stmts) if len(body_stmts) > 1 else body_stmts[0],
        SeqStmt(init_stmts) if len(init_stmts) > 1 else (init_stmts[0] if init_stmts else None),
        all_alloc_buffers,
        base_block.match_buffers,
        base_block.annotations,
    )
    
    return BlockRealize(base_iter_values, base_br.predicate, merged_block)


def _merge_fused_loops_in_func(mod: IRModule, func_name: str) -> IRModule:
    """
    对单个函数执行循环合并
    
    针对 reduction blocks 的处理步骤:
    1. 收集每个 block 的 inner_loops
    2. 对齐 inner_loops（通过 IR Transform 插入 For）
    3. 按 producer -> consumer 拓扑排序
    4. 按拓扑顺序一次性融合所有 block 并放到对应的 for 循环下
    """
    sch = tvm.tir.Schedule(mod)
    
    _merge_loops_by_bind(sch, func_name)
    fused_loop_rv = _merge_fused_top_loops(sch, func_name)
    
    sorted_inner_loops = []
    missing_loops_map = {}
    sorted_block_names = []
    
    if fused_loop_rv is not None:
        sorted_inner_loops, missing_loops_map, sorted_block_names = \
            _process_reduction_blocks(sch, fused_loop_rv)
    
    new_func = sch.mod[func_name]
    
    if missing_loops_map:
        new_func = _align_inner_loops_in_func(new_func, missing_loops_map)
    
    if len(sorted_block_names) >= 2:
        new_func = _merge_reduction_blocks_in_func(
            new_func, sorted_block_names, sorted_inner_loops
        )
    
    new_functions = {
        gvar: (new_func if gvar.name_hint == func_name else f)
        for gvar, f in sch.mod.functions.items()
    }
    return IRModule(new_functions)


@tvm.transform.module_pass(opt_level=0, name="MergeFusedLoops")
def MergeFusedLoops(mod: IRModule, ctx: tvm.transform.PassContext) -> IRModule:
    """TVM Pass: 合并带有 {"tag": "fused"} 注解的循环"""
    for gvar, func in mod.functions.items():
        if isinstance(func, PrimFunc):
            mod = _merge_fused_loops_in_func(mod, gvar.name_hint)
    return mod
