import tvm
from tvm import tir
from tvm.tir import PrimFunc, Stmt, For, BlockRealize
from tvm.ir import IRModule
from tvm.tir import functor, stmt_functor
from tvm.arith import Analyzer

@functor.mutator
class UnitLoopEliminator(functor.PyStmtExprMutator):

    def __init__(self):
        super().__init__()
        self.unit_loop_vars = dict()
        self.analyzer = Analyzer()

    def visit_for_(self, for_node: For) -> Stmt:
        is_unit_loop = isinstance(for_node.extent, tir.IntImm) and for_node.extent.value == 1
        
        if is_unit_loop:
            self.unit_loop_vars[for_node.loop_var] = for_node.min

        new_body = self.visit_stmt(for_node.body)

        if is_unit_loop:
            self.unit_loop_vars.pop(for_node.loop_var, None)
            return new_body
        else:
            if new_body is for_node.body:
                return for_node
            else:
                return For(
                    for_node.loop_var,
                    for_node.min,
                    for_node.extent,
                    for_node.kind,
                    new_body,
                    for_node.thread_binding,
                    for_node.annotations,
                )

    def visit_block_realize_(self, op: BlockRealize) -> Stmt:
        iter_map = dict()
        new_iter_vars = list()
        new_iter_values = list()
        for iter_var, iter_value in zip(op.block.iter_vars, op.iter_values):
            new_iter_value = self.analyzer.simplify(stmt_functor.substitute(iter_value, self.unit_loop_vars))
            if isinstance(new_iter_value, tir.IntImm):
                iter_map[iter_var.var] = new_iter_value
            else:
                new_iter_vars.append(iter_var)
                new_iter_values.append(new_iter_value)
        
        new_block = self.visit_stmt(op.block)
        
        new_block = stmt_functor.substitute(new_block, iter_map)
        
        new_block = tir.Block(
            new_iter_vars,
            new_block.reads,
            new_block.writes,
            new_block.name_hint,
            new_block.body,
            new_block.init,
            new_block.alloc_buffers,
            new_block.match_buffers,
            new_block.annotations,
        )
        
        return BlockRealize(new_iter_values, op.predicate, new_block)

def eliminate_unit_loops_in_func(func: PrimFunc) -> PrimFunc:
    eliminator = UnitLoopEliminator()
    new_body = eliminator.visit_stmt(func.body)

    if new_body is func.body:
        return func
    else:
        # 创建新的 PrimFunc
        return PrimFunc(
            func.params,
            new_body,
            func.ret_type,
            func.buffer_map,
            func.attrs,
            func.span,
        )


@tvm.transform.module_pass(opt_level=0, name="EliminateUnitLoops")
def EliminateUnitLoops(mod: IRModule, ctx: tvm.transform.PassContext) -> IRModule:
    new_functions = {}

    for global_var, base_func in mod.functions.items():
        if isinstance(base_func, PrimFunc):
            new_func = eliminate_unit_loops_in_func(base_func)
            new_functions[global_var] = new_func
        else:
            new_functions[global_var] = base_func

    return IRModule(new_functions)


def test_eliminate_unit_loops():
    """测试 eliminate_unit_loops pass"""
    import tvm.script
    from tvm.script import tir as T

    @tvm.script.ir_module
    class TestModule:
        @T.prim_func
        def main(A: T.Buffer((128, 64), "float32"), B: T.Buffer((128, 64), "float32")):
            T.func_attr({"global_symbol": "main"})
            for i in range(128):
                for j in range(1):  # unit loop
                    for k in range(64):
                        for l in range(1):  # unit loop
                            with T.block("compute"):
                                vi = T.axis.spatial(128, i)
                                vj = T.axis.spatial(1, j)
                                vk = T.axis.spatial(64, k)
                                vl = T.axis.spatial(1, l)
                                B[vi, vk] = A[vi, vk] * T.float32(2.0)

    print("=== 原始 IR ===")
    print(TestModule.script())

    with tvm.transform.PassContext(opt_level=0):
        transformed_mod2 = EliminateUnitLoops(TestModule)
    print("=== 应用EliminateUnitLoops Pass后 ===")
    print(transformed_mod2.script())


if __name__ == "__main__":
    test_eliminate_unit_loops()
