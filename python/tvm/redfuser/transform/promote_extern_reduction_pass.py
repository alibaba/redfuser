"""
PromoteExternReduction Pass

将te.extern得到的Block(For)处理成需要的形态

匹配策略：
  - 优先按block名字匹配
  - 当前支持: "tir_reduce_topk"
  - 后续可扩展其他类型
"""

from typing import List, Optional

import tvm
from tvm import tir
from tvm.tir import PrimFunc, Stmt, For, Block, BlockRealize, Evaluate, Call, IterVar
from tvm.tir import BufferLoad, BufferRegion, BufferStore, Broadcast
from tvm.ir import IRModule, Range


# 当前支持的block的名字
SUPPORTED_BLOCK_NAMES = {
    "tir_reduce_topk",
}


def _process_tir_reduce_topk_block(block: Block) -> Optional[Stmt]:
    # 1.结构匹配
    # 没有reads/writes
    if len(block.reads) != 0 or len(block.writes) != 0:
        return None

    # 理应是一个For
    body = block.body
    if not isinstance(body, For):
        return None

    # 提取嵌套的For
    for_chain = []
    cur = body
    while isinstance(cur, For):
        for_chain.append(cur)
        cur = cur.body

    # 最内层需要是Evaluate(Call)
    if not isinstance(cur, Evaluate):
        return None
    if not isinstance(cur.value, Call):
        return None
    if not cur.value.op.same_as(tir.op.Op.get("tir.vec_reduce")):
        return None
    
    evaluate = cur

    # 2.获取tir.vec_reduce函数内的信息
    call = evaluate.value
    # args: [op_type, num_topk, axis, input_load, *out_loads, reduce_var]
    num_topk = int(call.args[1])       # e.g. 8
    input_load = call.args[3]          # BufferLoad
    out_loads = list(call.args[4:-1])  # [values_buf_load, indices_buf_load]
    reduce_var = call.args[-1]         # Var(reduce循环变量)

    # 3.识别reduce轴
    reduce_for_idx = None
    for i, f in enumerate(for_chain):
        if f.loop_var.same_as(reduce_var):
            reduce_for_idx = i
            break

    if reduce_for_idx is None:
        # 找不到reduce轴,原样返回
        body = evaluate
        for f in reversed(for_chain):
            body = For(f.loop_var, f.min, f.extent, f.kind, body, annotations=f.annotations)
        return body

    # 4.创建block iter_vars
    block_iter_vars = []
    block_iter_values = []
    var_remap = {}  # old loop_var -> new block iter_var.var

    for i, f in enumerate(for_chain):
        iter_type = IterVar.CommReduce if i == reduce_for_idx else IterVar.DataPar
        new_name = f"v_{f.loop_var.name}"
        v = IterVar(Range(f.min, f.extent), new_name, iter_type)
        block_iter_vars.append(v)
        block_iter_values.append(f.loop_var)
        var_remap[f.loop_var] = v.var

    # 5.替换evaluate中的循环变量
    new_evaluate = tir.stmt_functor.substitute(evaluate, var_remap)

    # 6.构造reads
    input_indices_remapped = [
        var_remap.get(idx, idx) if isinstance(idx, tir.Var) else idx
        for idx in input_load.indices
    ]
    reads = [BufferRegion(input_load.buffer,
                          [Range.from_min_extent(idx, 1) for idx in input_indices_remapped])]

    # 7.构造writes
    writes = []
    for out_load in out_loads:
        out_regions = []
        for idx in out_load.indices:
            if isinstance(idx, tir.Ramp):
                out_regions.append(Range.from_min_extent(idx.base, idx.lanes))
            else:
                remapped = var_remap.get(idx, idx) if isinstance(idx, tir.Var) else idx
                out_regions.append(Range.from_min_extent(remapped, 1))
        writes.append(BufferRegion(out_load.buffer, out_regions))

    # 8.构造init
    init_stmts = []
    for out_load in out_loads:
        remapped_indices = []
        for idx in out_load.indices:
            if isinstance(idx, tir.Ramp):
                remapped_indices.append(idx)
            else:
                remapped_indices.append(var_remap.get(idx, idx) if isinstance(idx, tir.Var) else idx)
        
        # 构造广播初始化
        dtype = out_load.buffer.dtype # 这里目前应该永远都是float32
        if dtype == "int32":
            init_val = tvm.tir.const(-1, "int32")
        else:
            init_val = tvm.tir.const(-1e5, dtype)
        broadcast = Broadcast(init_val, num_topk)
        init_store = BufferStore(out_load.buffer, broadcast, remapped_indices)
        init_stmts.append(init_store)

    init_stmt = tir.SeqStmt(init_stmts) if len(init_stmts) > 1 else init_stmts[0]

    # 9.构造Block和BlockRealize
    new_block = Block(
        iter_vars=block_iter_vars,
        reads=reads,
        writes=writes,
        name_hint=block.name_hint,
        body=new_evaluate,
        init=init_stmt,
    )
    block_realize = BlockRealize(
        iter_values=block_iter_values,
        predicate=tvm.tir.const(True, "bool"),
        block=new_block,
    )

    # 10.用原来的For包裹
    result = block_realize
    for f in reversed(for_chain):
        result = For(f.loop_var, f.min, f.extent, f.kind, result, annotations=f.annotations)

    return result


PROCESS_FUNC = {
    "tir_reduce_topk": _process_tir_reduce_topk_block,
}


def _transform_block_realize(stmt: tir.BlockRealize) -> Optional[Stmt]:
    block = stmt.block

    if block.name_hint not in SUPPORTED_BLOCK_NAMES:
        return None

    return PROCESS_FUNC[block.name_hint](block)


def _promote_extern_reduction(mod: IRModule, func_name: str) -> IRModule:
    gvar = mod.get_global_var(func_name)
    func = mod[gvar]

    def _postorder(stmt):
        if isinstance(stmt, tir.BlockRealize):
            return _transform_block_realize(stmt)
        return None

    new_body = tir.stmt_functor.ir_transform(
        func.body, 
        preorder=None, 
        postorder=_postorder, 
        only_enable=["tir.BlockRealize"]
    )
    new_func = func.with_body(new_body)
    mod.update_func(gvar, new_func)

    return mod


@tvm.transform.module_pass(opt_level=0, name="PromoteExternReduction")
def PromoteExternReduction(mod: IRModule, ctx: tvm.transform.PassContext) -> IRModule:
    for gvar, func in mod.functions.items():
        if isinstance(func, PrimFunc):
            mod = _promote_extern_reduction(mod, gvar.name_hint)
    return mod
