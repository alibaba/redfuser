"""
GEMM Dtype Unify Pass - 统一 tl_gemm 输入 buffer 的数据类型

功能:
1. 检测每个 tl_gemm 调用中输入 buffer 的 dtype
2. 对于单个 tl_gemm，将其输入转换为相同的 dtype，优先级从高到低: fp16, fp32
3. 在 tl_gemm 调用之前插入 cast 语句
"""

import tvm
from tvm import tir
from tvm.tir import PrimFunc, Buffer
from tvm.ir import IRModule
from tvm.tir import functor
from typing import Dict, List, Tuple

buffer_to_tl_region = tvm.get_global_func("tir.Buffer2TL_Region")

# smaller value, higher priority
DTYPE_PRIORITY = {
    "float8_e4m3fn": 0,
    "float8_e4m3fnuz": 1,
    "float16": 2,
    "float32": 3,
}


def get_higher_priority_dtype(dtype1: str, dtype2: str) -> str:
    if DTYPE_PRIORITY.get(str(dtype1), 999) <= DTYPE_PRIORITY.get(str(dtype2), 999):
        return dtype1
    return dtype2


@functor.mutator
class GemmDtypeUnifyMutator(functor.PyStmtExprMutator):
    """
    对每个 tl_gemm 语句单独进行 dtype 统一处理。
    在 tl_gemm 之前插入必要的 cast 语句，并替换 tl_gemm 中的 buffer 引用。
    """

    def __init__(self):
        super().__init__()
        # 记录新分配的 buffer，用于后续添加到 alloc_buffers
        self.new_buffers: List[Buffer] = []
        # 已创建的 cast buffer 缓存: (原 buffer, 目标 dtype) -> 新 buffer
        self.cast_buffer_cache: Dict[Tuple[Buffer, str], Buffer] = {}
        # 计数器，用于生成唯一的 buffer 名称
        self.cast_counter = 0

    def _get_or_create_cast_buffer(self, old_buf: Buffer, target_dtype: str) -> Buffer:
        """获取或创建一个 cast buffer"""
        key = (old_buf, target_dtype)
        if key in self.cast_buffer_cache:
            return self.cast_buffer_cache[key]

        # 创建新 buffer
        dtype_suffix = target_dtype.replace("float", "fp")
        new_name = f"{old_buf.name}_{dtype_suffix}_{self.cast_counter}"
        self.cast_counter += 1

        new_buf = tir.decl_buffer(
            old_buf.shape,
            target_dtype,
            name=new_name,
            scope="local.fragment"
        )
        self.cast_buffer_cache[key] = new_buf
        self.new_buffers.append(new_buf)
        return new_buf

    def _create_cast_stmt(self, src_buf: Buffer, dst_buf: Buffer) -> tir.Stmt:
        """
        创建 cast 语句，使用 tl_copy 进行并行转换:
        T.tl_copy(T.tl_region(src_buf), T.tl_region(dst_buf))
        """
        src_region = buffer_to_tl_region(src_buf)
        dst_region = buffer_to_tl_region(dst_buf)
        return tir.Evaluate(tir.Call("void", tir.op.Op.get("tir.tl_copy"), [src_region, dst_region, tir.IntImm("int32", -1), tir.IntImm("int32", 0), tir.IntImm("int32", 0)]))

    def _replace_tl_region_buffer(self, call: tir.Call, new_buf: Buffer) -> tir.Call:
        """替换 tl_region 中的 buffer"""
        buffer_load = call.args[0]
        new_buffer_load = tir.BufferLoad(new_buf, buffer_load.indices)
        new_args = [new_buffer_load] + list(call.args[1:])
        return tir.Call(call.dtype, call.op, new_args)

    def visit_evaluate_(self, op: tir.Evaluate) -> tir.Stmt:
        """处理 Evaluate 语句，检测 tl_gemm 并进行转换"""
        tl_gemm_op = tir.op.Op.get("tir.tl_gemm")

        if not isinstance(op.value, tir.Call):
            return op

        call = op.value
        if not call.op.same_as(tl_gemm_op):
            return op

        # 提取 A, B buffer
        A_region = call.args[0]
        B_region = call.args[1]

        if not isinstance(A_region, tir.Call) or not isinstance(B_region, tir.Call):
            return op

        A_buf = A_region.args[0].buffer
        B_buf = B_region.args[0].buffer

        # 确定目标 dtype
        target_dtype = get_higher_priority_dtype(str(A_buf.dtype), str(B_buf.dtype))

        # 检查是否需要转换
        need_cast_A = str(A_buf.dtype) != target_dtype
        need_cast_B = str(B_buf.dtype) != target_dtype

        if not need_cast_A and not need_cast_B:
            # 无需转换
            return op

        # 收集需要插入的语句
        stmts_before: List[tir.Stmt] = []
        new_args = list(call.args)

        # 处理 A
        if need_cast_A:
            new_A_buf = self._get_or_create_cast_buffer(A_buf, target_dtype)
            cast_stmt = self._create_cast_stmt(A_buf, new_A_buf)
            stmts_before.append(cast_stmt)
            new_args[0] = self._replace_tl_region_buffer(A_region, new_A_buf)

        # 处理 B
        if need_cast_B:
            new_B_buf = self._get_or_create_cast_buffer(B_buf, target_dtype)
            cast_stmt = self._create_cast_stmt(B_buf, new_B_buf)
            stmts_before.append(cast_stmt)
            new_args[1] = self._replace_tl_region_buffer(B_region, new_B_buf)

        # 创建新的 tl_gemm 调用
        new_call = tir.Call(call.dtype, call.op, new_args)
        new_gemm_stmt = tir.Evaluate(new_call)

        # 返回 SeqStmt: cast 语句 + 新的 gemm 语句
        return tir.SeqStmt(stmts_before + [new_gemm_stmt])

    def _is_root_block(self, block: tir.Block) -> bool:
        """判断是否是 root block（没有 iter_vars 的顶层 block）"""
        return len(block.iter_vars) == 0

    def visit_block_realize_(self, op: tir.BlockRealize) -> tir.Stmt:
        """处理 BlockRealize，只在 root block 中添加新分配的 buffer"""
        block = op.block
        is_root = self._is_root_block(block)

        # 递归处理 body
        new_body = self.visit_stmt(block.body)

        # 检查是否有新的 buffer 被创建
        buffers_after = len(self.new_buffers)

        # 只在 root block 中添加所有新分配的 buffer
        if is_root and buffers_after > 0:
            new_alloc_buffers = list(block.alloc_buffers) + self.new_buffers
            new_block = tir.Block(
                block.iter_vars,
                block.reads,
                block.writes,
                block.name_hint,
                new_body,
                block.init,
                new_alloc_buffers,
                block.match_buffers,
                block.annotations,
            )
            return tir.BlockRealize(op.iter_values, op.predicate, new_block)

        if new_body.same_as(block.body):
            return op

        new_block = tir.Block(
            block.iter_vars,
            block.reads,
            block.writes,
            block.name_hint,
            new_body,
            block.init,
            block.alloc_buffers,
            block.match_buffers,
            block.annotations,
        )
        return tir.BlockRealize(op.iter_values, op.predicate, new_block)


def transform_gemm_dtype(func: PrimFunc) -> PrimFunc:
    """统一每个 tl_gemm 输入 buffer 的 dtype"""
    mutator = GemmDtypeUnifyMutator()
    new_body = mutator.visit_stmt(func.body)

    if new_body.same_as(func.body):
        return func

    return func.with_body(new_body)


@tvm.transform.module_pass(opt_level=0, name="UnifyGemmDtype")
def UnifyGemmDtype(mod: IRModule, ctx: tvm.transform.PassContext) -> IRModule:
    """
    Module Pass: 统一每个 tl_gemm 输入 buffer 的 dtype

    对于每个 tl_gemm 语句:
    1. 检测 A, B buffer 的 dtype
    2. 确定目标 dtype (优先级: fp16 > fp32)
    3. 在 tl_gemm 之前插入 tl_parallel cast 语句
    4. 替换 tl_gemm 中的 buffer 引用
    """
    new_functions = {}

    for gvar, func in mod.functions.items():
        if isinstance(func, PrimFunc):
            new_func = transform_gemm_dtype(func)
            new_functions[gvar] = new_func
        else:
            new_functions[gvar] = func

    return IRModule(new_functions)


if __name__ == "__main__":
    pass
