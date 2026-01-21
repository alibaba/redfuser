"""
IO Buffer Transform Pass - 为PrimFunc的输入输出创建buffer并插入数据搬运

功能:
1. 为PrimFunc的每个输入输出参数申请一个新的buffer
2. 创建load/store block，用于在global和buffer之间搬运数据
3. 替换函数体中的输入输出引用为新的buffer
"""

import tvm
from tvm import tir
from tvm.tir import Buffer, PrimFunc, Var, IterVar, BufferRegion
from tvm.ir import IRModule, Range
from typing import Dict, List, Set, Tuple
from tvm.tir import stmt_functor, functor
from tvm.tir.analysis import get_block_access_region, get_block_read_write_region


class IOBufferAnalyzer:
    """分析PrimFunc的输入输出buffer"""

    def __init__(self, func: PrimFunc):
        self.func = func

        # 输入buffer: buffer_var -> Buffer
        self.input_buffers: Dict[Var, Buffer] = {}

        # 输出buffer: buffer_var -> Buffer
        self.output_buffers: Dict[Var, Buffer] = {}

        # 所有block写入的buffer
        self.written_buffers: Set[Var] = set()

        # 输出buffer的索引列表
        self.output_buffer_indices: List[int] = []

        # 运行分析
        self._analyze()

    def _analyze(self):
        """分析函数的输入输出buffer"""
        # 第一步：收集所有参数buffer
        param_buffers = {}
        for idx, param in enumerate(self.func.params):
            if param in self.func.buffer_map:
                buffer = self.func.buffer_map[param]
                param_buffers[buffer.data] = (idx, buffer)

        # 第二步：分析哪些buffer被写入
        def visit(stmt):
            if isinstance(stmt, tir.BlockRealize):
                block = stmt.block
                for buffer_region in block.writes:
                    self.written_buffers.add(buffer_region.buffer.data)

        stmt_functor.post_order_visit(self.func.body, visit)

        # 第三步：根据是否被写入来判断输入/输出
        for buffer_var, (idx, buffer) in param_buffers.items():
            if buffer_var in self.written_buffers:
                # 被写入的是输出buffer
                self.output_buffers[buffer_var] = buffer
                self.output_buffer_indices.append(idx)
            else:
                # 未被写入的是输入buffer
                self.input_buffers[buffer_var] = buffer
        
        # print(f"\n=== IO分析结果 ===")
        # print(f"输入buffers: {[buf.name for buf in self.input_buffers.values()]}")
        # print(f"输出buffers: {[buf.name for buf in self.output_buffers.values()]}")
        # print(f"输出buffer索引: {self.output_buffer_indices}")

class IOBufferTransformer:
    """转换PrimFunc，为输入输出创建fragment buffer并插入搬运代码"""

    def __init__(self, func: PrimFunc, analyzer: IOBufferAnalyzer):
        self.func = func
        self.analyzer = analyzer

        # 旧buffer -> 新buffer的映射
        self.buffer_mapping: Dict[Var, Buffer] = {}

        # 生成新的buffer
        self._create_fragment_buffers()

    def _create_fragment_buffers(self):
        """为每个输入输出buffer创建对应的fragment buffer"""
        all_io_buffers = {**self.analyzer.input_buffers, **self.analyzer.output_buffers}

        for old_var, old_buffer in all_io_buffers.items():
            # 创建新的变量名
            new_name = f"{old_buffer.name}_fragment"

            # 创建新的buffer（shape和dtype与原buffer相同）
            new_buffer = tir.decl_buffer(
                old_buffer.shape, old_buffer.dtype, name=new_name
            )

            self.buffer_mapping[old_var] = new_buffer

            # print(f"创建fragment buffer: {old_buffer.name} -> {new_name}")

    def _create_load_store_stmt(
        self, src_buffer: Buffer, dst_buffer: Buffer, indices: List
    ) -> tir.Stmt:
        """创建从global加载到fragment的语句: dst[indices] = src[indices]"""
        load_expr = tir.BufferLoad(src_buffer, indices)
        store_stmt = tir.BufferStore(dst_buffer, load_expr, indices)
        return store_stmt

    def transform(self) -> PrimFunc:
        """执行转换"""
        # 创建IO语句插入器
        inserter = IOStmtInserter(
            self.buffer_mapping,
            self.analyzer.input_buffers,
            self.analyzer.output_buffers,
            self,
        )

        new_body = inserter.visit_stmt(self.func.body)

        new_attrs = dict(self.func.attrs) if self.func.attrs else {}
        new_attrs["out_idx"] = self.analyzer.output_buffer_indices

        return PrimFunc(
            self.func.params,
            new_body,
            self.func.ret_type,
            self.func.buffer_map,
            tvm.ir.make_node("ir.DictAttrs", **new_attrs),
            self.func.span
        )


@functor.mutator
class IOStmtInserter(functor.PyStmtExprMutator):
    """在现有block中插入IO语句的mutator"""

    def __init__(
        self,
        buffer_mapping: Dict[Var, Buffer],
        input_buffers: Dict[Var, Buffer],
        output_buffers: Dict[Var, Buffer],
        transformer: IOBufferTransformer,
    ):
        super().__init__()
        self.buffer_mapping = buffer_mapping
        self.input_buffers = input_buffers
        self.output_buffers = output_buffers
        self.transformer = transformer
        self.buffer_replacer = BufferReplacer(buffer_mapping)

    def visit_block_realize_(self, op: tir.BlockRealize) -> tir.Stmt:
        """访问BlockRealize，在block中插入IO语句"""
        block = op.block

        # 如果是root block，需要特殊处理
        # root block通常没有iter_vars或者iter_vars为空
        is_root_block = len(block.iter_vars) == 0

        if is_root_block:
            # 对于root block，直接递归处理其body
            new_body = self.visit_stmt(block.body)

            # 更新alloc_buffers，添加fragment buffers
            new_alloc_buffers = list(self.buffer_mapping.values()) + list(
                block.alloc_buffers
            )

            # 创建新的block
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
        else:
            # 对于普通block，需要在其body中插入IO语句
            # 获取block的循环变量
            block_vars = [iter_var.var for iter_var in block.iter_vars]

            # 收集输入buffer的访问索引
            reads_io_buffers = {}  # buffer_var -> indices
            for buf_region in block.reads:
                if buf_region.buffer.data in self.input_buffers:
                    # 从BufferRegion中提取索引 (min值)
                    indices = [r.min for r in buf_region.region]
                    reads_io_buffers[buf_region.buffer.data] = indices

            # 收集输出buffer的访问索引
            writes_io_buffers = {}  # buffer_var -> indices
            for buf_region in block.writes:
                if buf_region.buffer.data in self.output_buffers:
                    # 从BufferRegion中提取索引 (min值)
                    indices = [r.min for r in buf_region.region]
                    writes_io_buffers[buf_region.buffer.data] = indices

            # 创建load语句
            load_stmts = []
            for old_var, old_buffer in self.input_buffers.items():
                if old_var in reads_io_buffers:
                    new_buffer = self.buffer_mapping[old_var]
                    indices = reads_io_buffers[old_var]
                    load_stmt = self.transformer._create_load_store_stmt(
                        old_buffer, new_buffer, indices
                    )
                    load_stmts.append(load_stmt)

            # 创建store语句
            store_stmts = []
            for old_var, old_buffer in self.output_buffers.items():
                if old_var in writes_io_buffers:
                    new_buffer = self.buffer_mapping[old_var]
                    indices = writes_io_buffers[old_var]
                    store_stmt = self.transformer._create_load_store_stmt(
                        new_buffer, old_buffer, indices
                    )
                    store_stmts.append(store_stmt)

            # 替换原有body中的buffer引用
            original_body = self.buffer_replacer.visit_stmt(block.body)

            # 组合语句: load_stmts + original_body + store_stmts
            all_stmts = load_stmts + [original_body] + store_stmts

            if len(all_stmts) > 1:
                new_body = tir.SeqStmt(all_stmts)
            else:
                new_body = all_stmts[0] if all_stmts else original_body

            # 更新reads和writes
            rws_mapping = (
                {buffer.buffer.data: buffer.buffer for buffer in block.reads}
                | {buffer.buffer.data: buffer.buffer for buffer in block.writes}
                | {
                    self.buffer_mapping[buffer.buffer.data].data: self.buffer_mapping[
                        buffer.buffer.data
                    ]
                    for buffer in list(block.reads) + list(block.writes)
                    if buffer.buffer.data in self.buffer_mapping
                }
            )
            # 创建临时block
            new_block = tir.Block(
                block.iter_vars,
                [],
                [],
                block.name_hint,
                new_body,
                block.init,
                block.alloc_buffers,
                block.match_buffers,
                block.annotations,
            )
            rws = get_block_read_write_region(new_block, rws_mapping)
            # 去重：使用字典基于buffer.data去重
            reads_dict = {}
            for buf_region in rws[0]:
                reads_dict[buf_region.buffer.data] = buf_region

            writes_dict = {}
            for buf_region in rws[1]:
                writes_dict[buf_region.buffer.data] = buf_region

            # 从writes中移除reads中也有的（如果一个buffer既读又写，只保留在writes中）
            for key in writes_dict:
                if key in reads_dict:
                    del reads_dict[key]

            new_reads = list(reads_dict.values())
            new_writes = list(writes_dict.values())
            new_block = tir.Block(
                block.iter_vars,
                list(new_reads),
                list(new_writes),
                block.name_hint,
                new_body,
                block.init,
                block.alloc_buffers,
                block.match_buffers,
                block.annotations,
            )

            return tir.BlockRealize(op.iter_values, op.predicate, new_block)


@functor.mutator
class BufferReplacer(functor.PyStmtExprMutator):
    """替换stmt中的buffer引用"""

    def __init__(self, buffer_mapping: Dict[Var, Buffer]):
        super().__init__()
        self.buffer_mapping = buffer_mapping

    def visit_buffer_load_(self, op: tir.BufferLoad) -> tir.PrimExpr:
        """替换BufferLoad"""
        # 检查是否需要替换
        if op.buffer.data in self.buffer_mapping:
            new_buffer = self.buffer_mapping[op.buffer.data]
            # 访问indices
            new_indices = [self.visit_expr(idx) for idx in op.indices]
            return tir.BufferLoad(new_buffer, new_indices)

        # 否则继续递归访问
        new_indices = [self.visit_expr(idx) for idx in op.indices]
        if any(new_idx != old_idx for new_idx, old_idx in zip(new_indices, op.indices)):
            return tir.BufferLoad(op.buffer, new_indices)
        return op

    def visit_buffer_store_(self, op: tir.BufferStore) -> tir.Stmt:
        """替换BufferStore"""
        # 检查是否需要替换
        if op.buffer.data in self.buffer_mapping:
            new_buffer = self.buffer_mapping[op.buffer.data]
            new_value = self.visit_expr(op.value)
            new_indices = [self.visit_expr(idx) for idx in op.indices]
            return tir.BufferStore(new_buffer, new_value, new_indices)

        # 否则继续递归访问
        new_value = self.visit_expr(op.value)
        new_indices = [self.visit_expr(idx) for idx in op.indices]
        if new_value != op.value or any(
            new_idx != old_idx for new_idx, old_idx in zip(new_indices, op.indices)
        ):
            return tir.BufferStore(op.buffer, new_value, new_indices)
        return op


def transform_io_buffers(func: PrimFunc) -> PrimFunc:
    """转换函数，为输入输出创建fragment buffer"""

    # 步骤1: 分析输入输出
    analyzer = IOBufferAnalyzer(func)

    # 如果没有输入输出buffer，直接返回
    if not analyzer.input_buffers and not analyzer.output_buffers:
        print("没有输入输出buffer需要转换")
        return func

    # 步骤2: 执行转换
    transformer = IOBufferTransformer(func, analyzer)
    new_func = transformer.transform()

    return new_func


@tvm.transform.module_pass(opt_level=0, name="TransformIOBuffers")
def TransformIOBuffers(mod: IRModule, ctx: tvm.transform.PassContext) -> IRModule:
    """
    TVM Pass: 为PrimFunc的输入输出创建fragment buffer并插入数据搬运

    用法:
        pass_func = tvm.transform.Sequential([TransformIOBuffers])
        with tvm.transform.PassContext():
            mod = pass_func(mod)
    """
    new_functions = {}

    for gvar, func in mod.functions.items():
        if isinstance(func, PrimFunc):
            new_func = transform_io_buffers(func)
            new_functions[gvar] = new_func
        else:
            new_functions[gvar] = func

    return IRModule(new_functions)
