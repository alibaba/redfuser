"""
Transform Pass: Convert For loops with bind annotation to AttrStmt

功能:
1. 检测带有 annotations["bind"] 的 For 循环
2. 将这些 For 循环转换为 AttrStmt 节点（使用 IterVar + thread_extent）
3. 将 For 循环的 body 移出，作为 AttrStmt 的 body
"""

import tvm
from tvm import tir
from tvm.tir import PrimFunc
from tvm.ir import IRModule, Range
from tvm.tir import functor

@functor.mutator
class BlockIdxBinderMutator(functor.PyStmtExprMutator):
    """将带有 bind annotation 的 For 循环转换为 AttrStmt"""
    
    def __init__(self, func: PrimFunc):
        super().__init__()
        self.func = func

    def visit_for_(self, op: tir.For) -> tir.Stmt:
        # 递归处理 body
        new_body = self.visit_stmt(op.body)
        
        # 检查是否有 bind annotation
        if "bind" in op.annotations:
            thread_tag = str(op.annotations["bind"])  # "blockIdx.x/y/z"
            
            iter_var = tir.IterVar(
                dom=Range(op.min, op.extent),
                var=op.loop_var,
                iter_type=tir.IterVar.ThreadIndex,
                thread_tag=thread_tag,
            )
            
            attr_stmt = tir.AttrStmt(
                node=iter_var,
                attr_key="thread_extent",
                value=op.extent,
                body=new_body
            )
            
            return attr_stmt
        
        return tir.For(
            op.loop_var,
            op.min,
            op.extent,
            op.kind,
            new_body,
            op.thread_binding,
            op.annotations,
        )


def transform_bind_blockidx(func: PrimFunc) -> PrimFunc:
    """转换函数，将带有 bind annotation 的 For 循环转换为 AttrStmt"""
    
    mutator = BlockIdxBinderMutator(func)
    new_body = mutator.visit_stmt(func.body)
    
    # 创建新的 PrimFunc
    new_func = func.with_body(new_body)
    
    return new_func


@tvm.transform.module_pass(opt_level=0, name="BindBlockIdx")
def BindBlockIdx(mod: IRModule, ctx: tvm.transform.PassContext) -> IRModule:
    """Module pass: 转换所有 PrimFunc 中的 bind annotation"""
    new_functions = {}
    
    for gvar, func in mod.functions.items():
        if isinstance(func, PrimFunc):
            new_func = transform_bind_blockidx(func)
            new_functions[gvar] = new_func
        else:
            new_functions[gvar] = func
    
    return IRModule(new_functions)