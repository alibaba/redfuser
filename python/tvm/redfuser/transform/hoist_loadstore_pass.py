"""
Transform Pass: Hoist tl_copy with IO operands

功能:
1. 检测包含 tl_copy 调用的 Evaluate 节点，其中操作数涉及函数 IO
2. 根据 IO 操作数的索引，找到最近的 For 循环
3. 如果最近的 For 循环不是父节点，创建一个 block 包含 tl_copy 并放置在最近的 For 循环下
"""

import tvm
from tvm import tir
from tvm.tir import BufferRegion, PrimFunc
from tvm.ir import IRModule, Op
from typing import Dict, Set, List, Optional, Tuple
from tvm.tir import stmt_functor, functor

@functor.visitor
class TLCopyAnalyzer(functor.PyStmtExprVisitor):
    """分析 tl_copy 调用并收集相关信息"""

    def __init__(self, func: PrimFunc):
        super().__init__()
        self.func = func
        # 函数的 IO buffers (参数)
        self.io_buffers: Set = set()
        # 收集需要处理的 Evaluate 节点信息
        self.tl_copy_info: List[Dict] = []
        self.current_binding: Dict[tir.Var, tir.PrimExpr] = dict()
        # For 循环栈 (用于追踪嵌套的 For 循环)
        self.for_stack: List = []

    def analyze(self):
        # 收集所有 IO buffers
        for param in self.func.params:
            if param in self.func.buffer_map:
                buffer = self.func.buffer_map[param]
                self.io_buffers.add(buffer.data)

        self.visit_stmt(self.func.body)

    def visit_evaluate_(self, op: tir.Evaluate) -> None:
        io_index = self._is_tl_copy_with_io(op)
        if io_index == -1:
            return
        io_bindings = self._extract_io_bindings(op.value.args[io_index].args[0])
        nearest_for = self._find_nearest_for(io_bindings)
        if nearest_for and not nearest_for.same_as(self.for_stack[-1]):
            self.tl_copy_info.append({
                "evaluate": op,
                "nearest_for": nearest_for,
                "io_index": io_index,
                "io_bindings": io_bindings,
            })

    def visit_for_(self, op: tir.For) -> None:
        self.for_stack.append(op)
        self.visit_stmt(op.body)
        self.for_stack.pop()

    def visit_block_realize_(self, op: tir.BlockRealize) -> None:
        self.current_binding = {iter_var.var: (iter_var, iter_value) for iter_var, iter_value in zip(op.block.iter_vars, op.iter_values)}
        self.visit_stmt(op.block)
        self.current_binding = dict()

    def _is_tl_copy_with_io(self, evaluate: tir.Evaluate) -> int:
        """检查 Evaluate 是否包含 tl_copy 调用且操作数涉及 IO buffer"""
        if not isinstance(evaluate.value, tir.Call):
            return -1

        call = evaluate.value
        if call.op.name != "tir.tl_copy":
            return -1
        
        if self._is_io_tlregion(call.args[0]):
            return 0
        elif self._is_io_tlregion(call.args[1]):
            return 1
        return -1

    def _is_io_tlregion(self, expr: tir.PrimExpr) -> bool:
        """检查表达式中是否访问了 IO buffer"""
        if isinstance(expr, tir.Call) and expr.op.name == "tir.tl_region":
            buffer_load = expr.args[0]
            if isinstance(buffer_load, tir.BufferLoad):
                if buffer_load.buffer.data in self.io_buffers:
                    return True
        return False

    def _extract_io_bindings(self, buffer_load: tir.BufferLoad) -> Dict:
        bindings = dict()
        def _get_var(expr: tir.PrimExpr):
            if isinstance(expr, tir.Var):
                bindings[expr] = self.current_binding[expr]
        for index in buffer_load.indices:
            stmt_functor.post_order_visit(index, _get_var)
        return bindings

    def _get_inner_most_parallel_for(self):
        for for_idx, for_loop in enumerate(reversed(self.for_stack)):
            if for_loop.annotations.get("bind") is not None:
                return len(self.for_stack) - for_idx - 1
        return -1

    def _find_nearest_for(self, indices: Dict) -> Optional[tir.For]:
        """根据索引找到最近的 For 循环"""
        # 从内向外查找包含这些变量的最近 For 循环
        bindings = set([v[1] for v in indices.values()])
        inner_most_parallel_for_idx = self._get_inner_most_parallel_for()
        for for_idx, for_loop in enumerate(reversed(self.for_stack)):
            if for_loop.loop_var in bindings:
                nearest_for_idx = len(self.for_stack) - for_idx - 1
                if nearest_for_idx < inner_most_parallel_for_idx:
                    return self.for_stack[inner_most_parallel_for_idx]
                else: 
                    return self.for_stack[nearest_for_idx]

        return None


@functor.mutator
class TLCopyHoister(functor.PyStmtExprMutator):
    def __init__(self, func: PrimFunc, analyzer: TLCopyAnalyzer):
        super().__init__()
        self.func = func
        self.analyzer = analyzer
        self.tl_copy_info = analyzer.tl_copy_info

    def visit_for_(self, op: tir.For) -> tir.Stmt:
        # 递归处理循环体
        new_body = self.visit_stmt(op.body)

        if isinstance(new_body, tir.SeqStmt):
            new_body = list(new_body.seq)
        else:
            new_body = list([new_body])

        for info in self.tl_copy_info:
            if op.same_as(info["nearest_for"]):
                # 创建一个 Block 包含 tl_copy 和原有的 body
                io_index = info["io_index"]
                if io_index == 0:
                    block_name = f"load_{info['evaluate'].value.args[io_index].args[0].buffer.name}"
                    insert_loc = 0
                else:
                    block_name = f"store_{info['evaluate'].value.args[io_index].args[0].buffer.name}"
                    insert_loc = -1
                read_region = self._extract_buffer_region(info["evaluate"].value.args[0])
                write_region = self._extract_buffer_region(info["evaluate"].value.args[1])
                # 创建 Block
                block = tir.Block(
                    iter_vars=[v[0] for v in info["io_bindings"].values()],
                    reads=[read_region],
                    writes=[write_region],
                    name_hint=block_name,
                    body=info["evaluate"],
                    init=None,
                    alloc_buffers=[],
                    match_buffers=[],
                    annotations={},
                )

                # 创建 BlockRealize
                block_realize = tir.BlockRealize(
                    iter_values=[v[1] for v in info["io_bindings"].values()],
                    predicate=tir.const(True, "bool"),
                    block=block,
                )

                new_body.insert(insert_loc, block_realize)

        # 组合语句
        if len(new_body) > 1:
            new_body = tir.SeqStmt(new_body)
        else:
            new_body = new_body[0]
        # 返回新的 For 循环
        return tir.For(
            op.loop_var,
            op.min,
            op.extent,
            op.kind,
            new_body,
            op.thread_binding,
            op.annotations,
        )

    def visit_evaluate_(self, op: tir.Evaluate) -> tir.Stmt:
        """访问 Evaluate 节点"""
        for info in self.tl_copy_info:
            if op.same_as(info["evaluate"]):
                return tir.Evaluate(tir.const(0, "int32"))
        return op

    def _extract_buffer_region(self, op: tir.Call) -> tir.BufferRegion:
        buffer = op.args[0].buffer
        dom_min = op.args[0].indices
        dom_extent = op.args[1:]
        return BufferRegion(buffer, [tvm.ir.Range.from_min_extent(min, extent) for min, extent in zip(dom_min, dom_extent)])
            
def transform_hoist_tl_copy(func: PrimFunc) -> PrimFunc:
    """转换函数，重组 tl_copy 调用"""

    # 步骤1: 分析
    analyzer = TLCopyAnalyzer(func)
    analyzer.analyze()

    if not analyzer.tl_copy_info:
        return func

    hoister = TLCopyHoister(func, analyzer)
    new_body = hoister.visit_stmt(func.body)

    # 创建新的 PrimFunc
    new_func = func.with_body(new_body)

    return new_func


@tvm.transform.module_pass(opt_level=0, name="HoistTLCopy")
def HoistTLCopy(mod: IRModule, ctx: tvm.transform.PassContext) -> IRModule:
    new_functions = {}

    for gvar, func in mod.functions.items():
        if isinstance(func, PrimFunc):
            new_func = transform_hoist_tl_copy(func)
            new_functions[gvar] = new_func
        else:
            new_functions[gvar] = func

    return IRModule(new_functions)
