"""
Move Alloc Buffer Pass - 将 alloc_buffer 移动到最内层带有 bind annotation 的 for 循环中

功能:
1. 找到最内层带有 bind annotation 的 For 节点
2. 收集 Block 节点中的所有 alloc_buffers
3. 创建一个新的 Block，将收集的 buffers 移入其 alloc_buffers，并插入到最内层 For 的 body 中
"""

import tvm
from tvm import tir
from tvm.tir import PrimFunc, Buffer
from tvm.ir import IRModule
from tvm.tir import functor
from typing import List, Optional, Set


@functor.visitor
class InnermostBindForFinder(functor.PyStmtExprVisitor):
    """找到最内层带有 bind annotation 的 For 节点"""

    def __init__(self):
        super().__init__()
        self.innermost_bind_for: Optional[tir.For] = None
        self.current_bind_for: Optional[tir.For] = None

    def visit_for_(self, op: tir.For):
        # 检查是否有 bind annotation
        if "bind" in op.annotations:
            # 记录当前的 bind for
            prev_bind_for = self.current_bind_for
            self.current_bind_for = op
            # 更新最内层的 bind for
            self.innermost_bind_for = op
            # 递归访问 body
            self.visit_stmt(op.body)
            # 恢复
            self.current_bind_for = prev_bind_for
        else:
            # 没有 bind annotation，继续递归
            self.visit_stmt(op.body)


@functor.visitor
class AllocBufferCollector(functor.PyStmtExprVisitor):
    """收集所有 Block 节点中的 alloc_buffers"""

    def __init__(self):
        super().__init__()
        self.alloc_buffers: List[Buffer] = []
        self.blocks_with_alloc: List[tir.Block] = []

    def visit_block_(self, op: tir.Block):
        # 收集 alloc_buffers
        if op.alloc_buffers:
            self.alloc_buffers.extend(op.alloc_buffers)
            self.blocks_with_alloc.append(op)
        # 递归访问 body
        self.visit_stmt(op.body)


@functor.mutator
class AllocBufferMover(functor.PyStmtExprMutator):
    """将 alloc_buffers 从原来的 Block 中移除，并在最内层 bind for 中创建新的 Block"""

    def __init__(
        self,
        innermost_bind_for: tir.For,
        alloc_buffers: List[Buffer],
        blocks_to_clear: Set[tir.Block],
    ):
        super().__init__()
        self.innermost_bind_for = innermost_bind_for
        self.alloc_buffers = alloc_buffers
        self.blocks_to_clear = blocks_to_clear

    def visit_block_(self, op: tir.Block) -> tir.Block:
        # 递归处理 body
        new_body = self.visit_stmt(op.body)

        # 如果这个 block 在需要清除的列表中，移除其 alloc_buffers
        if op in self.blocks_to_clear:
            new_block = tir.Block(
                op.iter_vars,
                op.reads,
                op.writes,
                op.name_hint,
                new_body,
                op.init,
                [],  # 清空 alloc_buffers
                op.match_buffers,
                op.annotations,
            )
            return new_block

        # 否则保持原样
        if new_body.same_as(op.body):
            return op

        return tir.Block(
            op.iter_vars,
            op.reads,
            op.writes,
            op.name_hint,
            new_body,
            op.init,
            op.alloc_buffers,
            op.match_buffers,
            op.annotations,
        )

    def visit_for_(self, op: tir.For) -> tir.Stmt:
        # 递归处理 body
        new_body = self.visit_stmt(op.body)

        # 检查是否是最内层的 bind for
        if op.same_as(self.innermost_bind_for):
            wrapper_block = tir.Block(
                iter_vars=[],
                reads=[],
                writes=[],
                name_hint="tilelang_root",
                body=new_body,
                init=None,
                alloc_buffers=list(self.alloc_buffers),
                match_buffers=[],
                annotations={},
            )

            # 创建 BlockRealize
            wrapper_block_realize = tir.BlockRealize(
                iter_values=[],
                predicate=tir.const(True, "bool"),
                block=wrapper_block,
            )

            new_body = wrapper_block_realize

        # 返回新的 For 节点
        return tir.For(
            op.loop_var,
            op.min,
            op.extent,
            op.kind,
            new_body,
            op.thread_binding,
            op.annotations,
        )


def move_alloc_buffer_to_innermost_bind_for(func: PrimFunc) -> PrimFunc:
    """
    将 alloc_buffer 移动到最内层带有 bind annotation 的 for 循环中

    步骤:
    1. 找到最内层带有 bind annotation 的 For 节点
    2. 收集所有 Block 节点中的 alloc_buffers
    3. 创建一个新的 Block，将收集的 buffers 移入其 alloc_buffers，
       并插入到最内层 For 的 body 中
    """

    # 步骤 1: 找到最内层带有 bind annotation 的 For 节点
    finder = InnermostBindForFinder()
    finder.visit_stmt(func.body)

    if finder.innermost_bind_for is None:
        # 没有找到带有 bind annotation 的 For 节点，返回原函数
        print("Warning: No For node with bind annotation found")
        return func

    # print(f"Found innermost bind for: {finder.innermost_bind_for.loop_var}")

    # 步骤 2: 收集所有 Block 节点中的 alloc_buffers
    collector = AllocBufferCollector()
    collector.visit_stmt(func.body)

    if not collector.alloc_buffers:
        # 没有 alloc_buffers，返回原函数
        print("Warning: No alloc_buffers found")
        return func

    # print(f"Collected {len(collector.alloc_buffers)} alloc_buffers:")
    # for buf in collector.alloc_buffers:
    #     print(f"  - {buf.name}")

    # 步骤 3: 移动 alloc_buffers
    blocks_to_clear = set(collector.blocks_with_alloc)
    mover = AllocBufferMover(
        finder.innermost_bind_for, collector.alloc_buffers, blocks_to_clear
    )
    new_body = mover.visit_stmt(func.body)

    # 创建新的 PrimFunc
    new_func = func.with_body(new_body)

    return new_func


@tvm.transform.module_pass(opt_level=0, name="MoveAllocBuffer")
def MoveAllocBuffer(mod: IRModule, ctx: tvm.transform.PassContext) -> IRModule:
    """Module pass: 将所有 PrimFunc 中的 alloc_buffer 移动到最内层 bind for 中"""
    new_functions = {}

    for gvar, func in mod.functions.items():
        if isinstance(func, PrimFunc):
            new_func = move_alloc_buffer_to_innermost_bind_for(func)
            new_functions[gvar] = new_func
        else:
            new_functions[gvar] = func

    return IRModule(new_functions)
