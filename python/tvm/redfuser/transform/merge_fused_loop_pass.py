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
from tvm.tir import PrimFunc, For, Block, BlockRealize, SeqStmt, Stmt, functor
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


def _get_loop_signature(sch: tir.Schedule, loops: List[tir.schedule.LoopRV]) -> tuple:
    """获取循环列表的签名（循环数量和每个循环的 extent 值）"""
    sig = []
    for l in loops:
        loop = sch.get(l)
        # 使用 extent 的整数值作为签名（如果是常量）
        if hasattr(loop.extent, 'value'):
            sig.append(loop.extent.value)
        else:
            sig.append(str(loop.extent))
    return tuple(sig)


def _merge_child_loops(sch: tir.Schedule, fused_loop_rv: tir.schedule.LoopRV) -> None:
    """
    合并 fused loop 内部名字以 "reduction" 开头的 block 和它的 consumer blocks（具有相同循环结构的）
    
    使用 sch.get_consumers() 获取每个 block 的 consumers，
    如果 consumer 具有相同的内层循环结构，则合并它们的循环。
    """
    child_blocks = sch.get_child_blocks(fused_loop_rv)
    if len(child_blocks) < 2:
        return
    
    # 使用 block name 作为 key（因为 BlockRV 会变化）
    # 收集每个 block 的信息：name -> (block_rv, inner_loops)
    block_info: List[tuple] = []  # [(name, block_rv, inner_loops), ...]
    
    for block_rv in child_blocks:
        block_name = sch.get(block_rv).name_hint
        # 只处理名字以 "reduction" 开头的 blocks
        if not block_name.startswith("reduction"):
            continue
        
        block_loops = list(sch.get_loops(block_rv))
        for i, loop in enumerate(block_loops):
            if sch.get_sref(loop).same_as(sch.get_sref(fused_loop_rv)):
                inner_loops = block_loops[i + 1:]
                block_info.append((block_name, block_rv, inner_loops))
                break
    
    if len(block_info) < 2:
        return
    
    # 为每个 block 计算循环签名，使用 name 作为 key
    block_signatures: Dict[str, tuple] = {}
    block_inner_loops: Dict[str, list] = {}
    name_to_rv: Dict[str, tir.schedule.BlockRV] = {}
    
    for name, block_rv, inner_loops in block_info:
        block_signatures[name] = _get_loop_signature(sch, inner_loops)
        block_inner_loops[name] = inner_loops
        name_to_rv[name] = block_rv
    
    # 收集需要合并的组：block 和它的同签名 consumers
    merged_names = set()
    groups_to_merge = []
    
    for name, block_rv, _ in block_info:
        if name in merged_names:
            continue
        
        block_sig = block_signatures.get(name)
        if block_sig is None:
            continue
        
        # 使用 sch.get_consumers 获取 consumer blocks
        consumer_blocks = sch.get_consumers(block_rv)
        
        # 找到具有相同签名的 consumers（且名字以 reduction 开头）
        same_sig_consumer_names = []
        for consumer in consumer_blocks:
            consumer_name = sch.get(consumer).name_hint
            if consumer_name not in merged_names \
               and consumer_name.startswith("reduction") \
               and block_signatures.get(consumer_name) == block_sig:
                same_sig_consumer_names.append(consumer_name)
        
        if same_sig_consumer_names:
            # 将 block 和它的同签名 consumers 组成一组
            group_names = [name] + same_sig_consumer_names
            groups_to_merge.append(group_names)
            merged_names.update(group_names)
    
    # 对每组 blocks 进行循环合并
    for group_names in groups_to_merge:
        inner_loops_list = [block_inner_loops[n] for n in group_names]
        if not inner_loops_list[0]:  # 没有内层循环
            continue
        
        # 合并最内层loop
        loops_to_merge = [loops[-1] for loops in inner_loops_list]
        if len(loops_to_merge) >= 2:
            sch.merge(*loops_to_merge)


@functor.mutator
class BlockMerger(functor.PyStmtExprMutator):
    """合并带有 fused tag 的循环内部共享同一内层循环的 blocks"""
    
    def __init__(self):
        super().__init__()
        self.in_fused_loop = False
    
    def visit_for_(self, op: For) -> Stmt:
        is_fused = op.annotations.get("tag") == "fused"
        
        if is_fused and not self.in_fused_loop:
            self.in_fused_loop = True
            new_body = self._process_fused_loop_body(op.body)
            self.in_fused_loop = False
            
            if new_body is not op.body:
                return For(op.loop_var, op.min, op.extent, op.kind,
                          new_body, op.thread_binding, op.annotations)
            return op
        
        new_body = self.visit_stmt(op.body)
        if new_body is op.body:
            return op
        return For(op.loop_var, op.min, op.extent, op.kind,
                  new_body, op.thread_binding, op.annotations)
    
    def _process_fused_loop_body(self, stmt: Stmt) -> Stmt:
        """处理 fused loop 的 body，只合并共享同一内层循环的 blocks"""
        if isinstance(stmt, SeqStmt):
            # 处理 SeqStmt 中的每个子语句
            new_seq = []
            for s in stmt.seq:
                processed = self._process_fused_loop_body(s)
                new_seq.append(processed)
            
            # 检查是否有变化
            if all(new_seq[i] is stmt.seq[i] for i in range(len(new_seq))):
                return stmt
            return SeqStmt(new_seq)
        
        elif isinstance(stmt, For):
            # 收集这个 For 循环下的所有 blocks
            blocks = []
            self._collect_blocks_under_loop(stmt.body, blocks)
            
            if len(blocks) >= 2:
                # 合并这些 blocks
                merged_block_realize = self._merge_block_group(blocks, stmt)
                # 重建循环结构
                return For(stmt.loop_var, stmt.min, stmt.extent, stmt.kind,
                          merged_block_realize, stmt.thread_binding, stmt.annotations)
            else:
                # 递归处理内层
                new_body = self._process_fused_loop_body(stmt.body)
                if new_body is stmt.body:
                    return stmt
                return For(stmt.loop_var, stmt.min, stmt.extent, stmt.kind,
                          new_body, stmt.thread_binding, stmt.annotations)
        
        elif isinstance(stmt, BlockRealize):
            return stmt
        
        return stmt
    
    def _collect_blocks_under_loop(self, stmt: Stmt, blocks: List[BlockRealize]) -> None:
        """收集循环下直接的 blocks（不递归进入内层循环）"""
        if isinstance(stmt, BlockRealize):
            blocks.append(stmt)
        elif isinstance(stmt, SeqStmt):
            for s in stmt.seq:
                self._collect_blocks_under_loop(s, blocks)
        # 不递归进入内层 For 循环
    
    def _merge_block_group(self, blocks: List[BlockRealize], parent_loop: For) -> Stmt:
        """合并一组共享同一父循环的 blocks"""
        if len(blocks) == 1:
            return blocks[0]
        
        first_br = blocks[0]
        first_block = first_br.block
        first_iter_vars = first_block.iter_vars
        
        # 构建变量替换映射
        var_map: Dict[tir.Var, tir.PrimExpr] = {}
        for br in blocks[1:]:
            for iv1, iv2 in zip(first_iter_vars, br.block.iter_vars):
                var_map[iv2.var] = iv1.var
        
        # 收集并合并内容
        body_stmts, init_stmts = [], []
        all_reads, all_writes = [], []
        all_alloc_buffers = []
        
        for idx, br in enumerate(blocks):
            block = br.block
            if idx == 0:
                body_stmts.append(block.body)
                if block.init:
                    init_stmts.append(block.init)
                all_reads.extend(block.reads)
                all_writes.extend(block.writes)
            else:
                body_stmts.append(tir.stmt_functor.substitute(block.body, var_map))
                if block.init:
                    init_stmts.append(tir.stmt_functor.substitute(block.init, var_map))
                all_reads.extend(self._substitute_buffer_region(r, var_map) for r in block.reads)
                all_writes.extend(self._substitute_buffer_region(w, var_map) for w in block.writes)
            all_alloc_buffers.extend(block.alloc_buffers)
        
        # 去重 reads/writes
        reads_dict = {r.buffer.data: r for r in all_reads}
        writes_dict = {w.buffer.data: w for w in all_writes}
        
        merged_block = Block(
            first_iter_vars,
            list(reads_dict.values()),
            list(writes_dict.values()),
            first_block.name_hint + "_merged",
            SeqStmt(body_stmts) if len(body_stmts) > 1 else body_stmts[0],
            SeqStmt(init_stmts) if len(init_stmts) > 1 else (init_stmts[0] if init_stmts else None),
            all_alloc_buffers,
            first_block.match_buffers,
            first_block.annotations,
        )
        
        return BlockRealize(first_br.iter_values, first_br.predicate, merged_block)
    
    def _substitute_buffer_region(self, buf_region: tir.BufferRegion, 
                                   var_map: Dict[tir.Var, tir.PrimExpr]) -> tir.BufferRegion:
        """替换 BufferRegion 中的变量"""
        new_region = [
            tvm.ir.Range.from_min_extent(
                tir.stmt_functor.substitute(rng.min, var_map),
                tir.stmt_functor.substitute(rng.extent, var_map)
            ) for rng in buf_region.region
        ]
        return tir.BufferRegion(buf_region.buffer, new_region)


def _merge_blocks_in_func(func: PrimFunc) -> PrimFunc:
    """使用 BlockMerger 合并函数中 fused loop 内的 blocks"""
    merger = BlockMerger()
    new_body = merger.visit_stmt(func.body)
    if new_body is func.body:
        return func
    return PrimFunc(func.params, new_body, func.ret_type, func.buffer_map, func.attrs, func.span)


def _merge_fused_loops_in_func(mod: IRModule, func_name: str) -> IRModule:
    """对单个函数执行循环合并"""
    sch = tvm.tir.Schedule(mod)
    
    _merge_loops_by_bind(sch, func_name)
    fused_loop_rv = _merge_fused_top_loops(sch, func_name)
    if fused_loop_rv is not None:
        _merge_child_loops(sch, fused_loop_rv)
    new_func = _merge_blocks_in_func(sch.mod[func_name])
    
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
