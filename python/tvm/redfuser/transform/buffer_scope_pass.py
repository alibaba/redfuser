import tvm
from tvm import tir
from tvm.tir import PrimFunc
from tvm.ir import IRModule
from typing import Dict, Set
from .common_analysis_v2 import (
    normalize_prim_func,
    BlockInfo,
    get_buffer_load_from_prim_expr,
    is_gemm_stmt,
)


class BufferScopeAnalyzer:
    def __init__(self, gvar: tvm.ir.GlobalVar, func: PrimFunc):
        self.func = func
        mod = IRModule(functions={gvar: func})
        self.sch = tir.Schedule(mod)
        self.sch.work_on(gvar.name_hint)
        self.blocks_info = normalize_prim_func(self.sch)
        # buffer -> set of scopes
        self.buffer_scopes: Dict[tir.Buffer, Set[str]] = {}
        self.func_params = set()

        for param in func.params:
            if param in func.buffer_map:
                buffer = func.buffer_map[param]
                self.func_params.add(buffer)

        for blk_info in self.blocks_info:
            self._visit_block(blk_info)

    def _visit_block(self, block_info: BlockInfo):
        
        def _is_pure_load_stmt(stmt: tir.Stmt) -> bool:
            if not isinstance(stmt, tir.BufferStore):
                return False
            # alloc buffer = global buffer
            if isinstance(stmt.value, tir.BufferLoad) and stmt.value.buffer in self.func_params:
                return True
            # global buffer = alloc buffer
            if isinstance(stmt.value, tir.BufferLoad) and stmt.buffer in self.func_params:
                return True
            return False

        def _visit_stmt(stmt: tir.Stmt):
            if isinstance(stmt, tir.BufferStore):
                if _is_pure_load_stmt(stmt):
                    return
                self._register_buffer_usage(stmt.buffer, "fragment")
                if is_gemm_stmt(stmt):
                    load_buffers = get_buffer_load_from_prim_expr(stmt.value.b)
                    for load_buffer in load_buffers:
                        self._register_buffer_usage(load_buffer.buffer, "shared")
                else:
                    load_buffers = get_buffer_load_from_prim_expr(stmt.value)
                    for load_buffer in load_buffers:
                        self._register_buffer_usage(load_buffer.buffer, "fragment")
            elif isinstance(stmt, tir.Evaluate):
                if stmt.value.op.same_as(tir.op.Op.get("tir.vec_reduce")):
                    op_type, vec_len, axis = stmt.value.args[0], stmt.value.args[1], stmt.value.args[2]
                    if op_type == "topk":
                        self._register_buffer_usage(stmt.value.args[3].buffer, "fragment")
                        self._register_buffer_usage(stmt.value.args[4].buffer, "fragment")
                        self._register_buffer_usage(stmt.value.args[5].buffer, "fragment")
                else:
                    raise ValueError(f"Invalid statement: {stmt}")

        block = self.sch.get(block_info.block_rv)
        if isinstance(block.body, tir.SeqStmt):
            for stmt in block.body:
                _visit_stmt(stmt)
        elif isinstance(block.body, tir.BufferStore):
            _visit_stmt(block.body)
        else:
            raise ValueError(f"Invalid block body: {block.body}")

    def _register_buffer_usage(self, buffer: tir.Buffer, scope: str):
        if buffer in self.func_params:
            return

        if buffer not in self.buffer_scopes:
            self.buffer_scopes[buffer] = set()

        self.buffer_scopes[buffer].add(scope)

    def get_final_scope(self, buffer: tir.Buffer) -> str:
        scopes = self.buffer_scopes[buffer]

        if "fragment" in scopes:
            return "local.fragment"

        if "shared" in scopes:
            return "shared"

        raise ValueError(f"Buffer {buffer.name_hint} has no scope")


def transform_buffer_scope(gvar: tvm.ir.GlobalVar, func: PrimFunc) -> PrimFunc:
    """转换函数中buffer的scope"""

    # 步骤1: 分析buffer使用情况
    analyzer = BufferScopeAnalyzer(gvar, func)

    # print("\n=== Buffer分析结果 ===")
    # for buf_var, scopes in analyzer.buffer_scopes.items():
    #     final_scope = analyzer.get_final_scope(buf_var)
    #     print(f"Buffer '{buf_var.name}': scopes={scopes}, final_scope={final_scope}")

    # 步骤2: 转换IR
    for blk in analyzer.blocks_info:
        block = analyzer.sch.get(blk.block_rv)
        writes = [w_buf.buffer for w_buf in block.writes]
        for buffer, scopes in analyzer.buffer_scopes.items():
            scope = analyzer.get_final_scope(buffer)
            for i, write in enumerate(writes):
                if buffer == write:
                    analyzer.sch.set_scope(blk.block_rv, i, scope)
                    break

    return analyzer.sch.mod.functions[gvar]


@tvm.transform.module_pass(opt_level=0, name="SetBufferScope")
def SetBufferScope(mod: IRModule, ctx: tvm.transform.PassContext) -> IRModule:
    new_functions = {}

    for gvar, func in mod.functions.items():
        if isinstance(func, PrimFunc):
            new_func = transform_buffer_scope(gvar, func)
            new_functions[gvar] = new_func
        else:
            new_functions[gvar] = func

    return IRModule(new_functions)
