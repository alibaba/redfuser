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
            if not isinstance(stmt, tir.BufferStore):
                raise ValueError(f"Invalid statement: {stmt}")
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

def test_complex_case():
    """测试更复杂的情况：buffer在多种类型的block中使用"""
    from tvm.script import tir as T

    @tvm.script.ir_module
    class ComplexModule:
        @T.prim_func
        def main(
            A: T.Buffer((128, 128), "float32"),
            B: T.Buffer((128, 128), "float32"),
            C: T.Buffer((128, 128), "float32"),
        ):
            T.func_attr({"global_symbol": "main"})

            temp1 = T.alloc_buffer((128, 128), "float32")
            temp2 = T.alloc_buffer((128, 128), "float32")
            A_local = T.alloc_buffer((128, 128), "float32")
            B_local = T.alloc_buffer((128, 128), "float32")
            C_local = T.alloc_buffer((128, 128), "float32")

            # Gemm block - temp1应该是shared
            for i, j, k in T.grid(128, 128, 128):
                with T.block("gemm"):
                    vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                    with T.init():
                        temp1[vi, vj] = T.float32(0)
                    A_local[vi, vk] = A[vi, vk]
                    B_local[vk, vj] = B[vk, vj]
                    temp1[vi, vj] = temp1[vi, vj] + A_local[vi, vk] * B_local[vk, vj]

            # Add block - temp2应该是fragment
            for i, j in T.grid(128, 128):
                with T.block("add"):
                    vi, vj = T.axis.remap("SS", [i, j])
                    temp2[vi, vj] = temp1[vi, vj] + T.float32(1.0)

            # ReLU block - temp2应该保持fragment
            for i, j in T.grid(128, 128):
                with T.block("relu"):
                    vi, vj = T.axis.remap("SS", [i, j])
                    C_local[vi, vj] = T.max(temp2[vi, vj], T.float32(0))
                    C[vi, vj] = C_local[vi, vj]

    print("\n\n=== 测试复杂情况 ===")
    print("=== 原始IR ===")
    print(ComplexModule.script())

    try:
        pass_func = tvm.transform.Sequential([SetBufferScope])
        with tvm.transform.PassContext(opt_level=0):
            mod = pass_func(ComplexModule)

        print("\n=== 应用BufferScope Pass后 ===")
        print(mod.script())
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    test_complex_case()
