"""
TIR Online Fusion Generator

将级联的规约Block转换为Online算法形式,每个Block包含4步:
1. 更新prev_reduce
2. 计算rescale_factor和归约输入
3. do_rescale
4. do_reduce

特别的:
第一个reduce(不依赖其他的reduce结果),不需要计算rescale_factor和do_rescale
最后一个reduce(规约结果不被其他规约使用),不需要更新prev_reduce

生成的多个Block会在后续pass中进行真正的循环融合
"""
from typing import Optional

import tvm
from tvm import tir

from functools import reduce
import sympy as sp

from .tir_reduction_analyzer import CascadedGroupInfo
from .utils import BMat, ElementwiseApplyNAry
from .common_analysis_v2 import _get_blockrealize


def transform_single_block_with_split(
    idx: int,
    reduction_infos: CascadedGroupInfo,
    reduce_funcs_list: list[tuple[BMat | None, BMat | None, BMat | None]],
    pro_ep_map: dict,
    num_split: Optional[int]
):
    # 这个函数默认num_split>0,所以后面不会做num_split相关的判断
    assert num_split is not None and num_split > 0

    # 好像只需要替换嵌套For下面最小的那个block,但是分配的buffer需要在最大的block上添加才行
    sch = reduction_infos.sch
    block_info = reduction_infos.cascaded_group[idx]
    block_rv = block_info.block_rv
    block = sch.get(block_rv)
    reduction_config = reduction_infos.reduction_configs[idx]

    # 处理循环相关(循环变量,索引等?)
    # 在这里得到所有的循环变量(block.iter_vars[x].var),后续在该block内生成的所有BufferLoad和BufferStore都应该使用这里的索引变量
    # 实现方式是:要产生的Load或者Store,会得到其对应的indices的字符串,再由一个dict映射得到该Block内的索引变量
    # 这是否是一个糟糕的设计?(这要求几乎所有Block的索引变量都是有意义的,尽管我们在te表达式中已经在强调这一点)
    # 注:这个map的value的类型是tvm.tir.Var,iv的类型是tvm.tir.IterVar
    iter_vars_map = {iv.var.name : iv.var for iv in block.iter_vars}
    # 手动添加split变量(还是等后面的loop构造出来之后再加?)
    iter_vars_map["v_split"] = tvm.tir.Var("v_split", "int32")

    # 传入某一个BufferLoad或者BufferStore的indices,将其转换成全部由该Block的iter_vars_map的indices数组
    # 如果转换出来前后的长度不对,那错误应该在外部产生
    def _indices_to_indices(indices: list[tir.Var]):
        name_indices = [v.name for v in indices]
        out_indices = list(map(lambda name: iter_vars_map[name], name_indices))

        assert len(out_indices) == len(indices), f"{len(out_indices)=} != {len(indices)=}"

        return out_indices

    # 1. 判断当前block的consumers,若consumers中存在属于当前cascaded_group的block,那么需要计算prev_reduce
    def _need_update_prev_reduce(block_rv): # BlockRV
        consumers = sch.get_consumers(block_rv) # [BlockRV]
        # BlockRV -> Block,然后比较
        for consumer in consumers:
            if sch.get(consumer) in [sch.get(block_info.block_rv) for block_info in reduction_infos.cascaded_group]:
                return True
        return False

    # 产生prev_reduce_buffer变量和更新prev_reduce的stmt
    # reduce_target_buffer_load会在后面用到,所以放在外面构建
    reduce_target_buffer_load = reduction_infos.y_buffer_load_map[reduction_config.reduce_target]
    reduce_target_buffer = reduce_target_buffer_load.buffer
    
    need_update_prev_reduce = _need_update_prev_reduce(block_rv)
    if need_update_prev_reduce:
        prev_reduce_buffer = tir.decl_buffer(
            shape=reduce_target_buffer.shape,
            dtype=reduce_target_buffer.dtype,
            name=f"prev_{reduce_target_buffer.name}"
        )

        # prev更新的就是该block的上一次循环的reduce_target的结果
        prev_reduce_stmt = tir.BufferStore(
            buffer=prev_reduce_buffer,
            value=tir.BufferLoad(reduce_target_buffer, _indices_to_indices(reduce_target_buffer_load.indices)),
            indices=_indices_to_indices(reduce_target_buffer_load.indices)
        )

        # 在这里更新prev_buffer_load_map,目的是为下面从sympy表达式产生tir时,能够找到prev_xx对应的BufferLoad
        reduction_infos.prev_buffer_load_map[f"prev_{reduction_config.reduce_target}"] = tvm.tir.BufferLoad(buffer=prev_reduce_buffer, indices=reduce_target_buffer_load.indices)
    else:
        prev_reduce_buffer = None
        prev_reduce_stmt = None
    
    # 2.1 判断当前block的producers
    # 若所有的producers都不来自当前cascaded_group,则不需要计算resacle_factor和do_resacle
    def _need_cal_rescale_factor(block_rv): # BlockRV
        producers = sch.get_producers(block_rv) # [BlockRV]
        # BlockRV -> Block,然后比较
        for producer in producers:
            if sch.get(producer) in [sch.get(block_info.block_rv) for block_info in reduction_infos.cascaded_group]:
                return True
        return False

    def sympy_expr_to_tir(expr, ew_map: dict):
        if isinstance(expr, BMat):
            return sympy_expr_to_tir(expr.expr, ew_map)
        
        if isinstance(expr, ElementwiseApplyNAry):
            # 填充ew_map, e.g. {"d0": "y0"}, y0可由y_map再映射到对应的实际Buffer
            variables = expr.function.variables
            operands = expr.operands # operands可能会是复杂的表达式吗?
            assert len(variables) == len(operands)

            for (var, op) in zip(variables, operands):
                ew_map[var.name] = op.expr.name

            return sympy_expr_to_tir(expr.function.expr, ew_map)
        
        if isinstance(expr, sp.functions.elementary.exponential.exp):
            return tir.exp(sympy_expr_to_tir(expr.exp, ew_map))
        
        if expr.is_Mul:
            return reduce(lambda a, b: tir.Mul(a, b), [sympy_expr_to_tir(arg, ew_map) for arg in expr.args])

        if expr.is_Add:
            return reduce(lambda a, b: tir.Add(a, b), [sympy_expr_to_tir(arg, ew_map) for arg in expr.args]) 

        if expr.is_Pow:
            return reduce(lambda a, b: tir.pow(a, b), [sympy_expr_to_tir(arg, ew_map) for arg in expr.args])

        # d0, d1 ...
        if expr.is_Dummy:
            xyc = ew_map[expr.name]
            if xyc.startswith("x"):
                buffer_load = reduction_infos.x_buffer_load_map[xyc]
            elif xyc.startswith("y"):
                buffer_load = reduction_infos.y_buffer_load_map[xyc]
            elif xyc.startswith("c"): # 这个if应该不会进来,因为c不是Dummy,而是Symbol
                buffer_load = reduction_infos.c_buffer_load_map[xyc]
            elif xyc.startswith("prev"):
                buffer_load = reduction_infos.prev_buffer_load_map[xyc]
            else:
                buffer_load = None

            return tir.BufferLoad(buffer_load.buffer, _indices_to_indices(buffer_load.indices))

        # d0 -> -1 * d0
        # TYPE CHECK!
        if expr.is_Number:
            # FIXME 所有reduce操作的结果都在TE表达式中被cast到了float32...
            return tir.FloatImm("float32", float(expr))

        raise NotImplementedError(f"Unknown expr type: {type(expr)}")

    need_cal_rescale_factor = _need_cal_rescale_factor(block_rv)
    if need_cal_rescale_factor:
        H_expr = reduce_funcs_list[idx][1]
        prev_H_expr = reduce_funcs_list[idx][2]
        # TODO 如果是max的话,这里应该要做减法
        rescale_factor_expr = H_expr / prev_H_expr

        ew_map = {}
        rescale_factor_value = sympy_expr_to_tir(rescale_factor_expr, ew_map)

        # 从ew_map的value中随便取一个出来,默认取第0个,找到对应的BufferLoad,rescale factor的shape和dtype应该和随机取出来的这个变量保持一致
        # 命名则以rescale_factor为前缀,idx作为序号标识符
        ew_buffer_load = reduction_infos.y_buffer_load_map[list(ew_map.values())[0]] if list(ew_map.values())[0].startswith('y') else reduction_infos.prev_buffer_load_map[list(ew_map.values())[0]]
        rescale_factor_buffer = tir.decl_buffer(
            shape=ew_buffer_load.buffer.shape,
            dtype=ew_buffer_load.buffer.dtype,
            name=f"rescale_factor_{idx}"
        )

        rescale_factor_stmt = tir.BufferStore(
            buffer=rescale_factor_buffer,
            value=rescale_factor_value,
            indices=_indices_to_indices(ew_buffer_load.indices)
        )
    else:
        rescale_factor_buffer = None
        rescale_factor_stmt = None

    # 2.2 产生输入数据
    ori_reduce_func = reduction_infos.reduce_funcs[idx]

    def _remove_cast(e, dtype_stack: list):
        if isinstance(e, tir.Cast):
            dtype_stack.append(e.dtype)
            return _remove_cast(e.value, dtype_stack)
        return e

    def _replace_sub(e, sub, load, depth):
        if isinstance(e, tir.Cast) and depth > 0:
            return tir.Cast(e.dtype, _replace_sub(e.value, sub, load, depth - 1))
        
        if tvm.ir.structural_equal(e, sub):
            return load
        else:
            return e

    def _collect_sub_func_loads(e):
        if isinstance(e, tir.BufferLoad):
            sub_reduce_func_loads.append(e)

    input_buffers = []
    input_buffer_stmts = []

    """
        输入的产生逻辑:
        1. 对于GEMM op,需要处理ori_reduce_func左右两边的表达式,最终得到的replaced_reduce_func有两种情况:
            a. y = y + a * b
            b. y = y + T.Cast("?", a) * T.Cast("?", b)
            也即replaced_reduce_func允许至多一个T.Cast

        2. 对于非GEMM op,期望得到的replaced_reduce_func中不存在显式的T.Cast,也即可以直接用ori_reduce_func来构造输入
    """
    if block_info.is_gemm():
        assert isinstance(ori_reduce_func, tir.Mul)

        ori_reduce_funcs = [ori_reduce_func.a, ori_reduce_func.b]

        # 同时开始便利a和b,找到能够同时剥离Cast的最大深度,该位置后的value即为需要提取成为input的value
        dtype_stack_a = []
        dtype_stack_b = []

        _ = _remove_cast(ori_reduce_funcs[0], dtype_stack_a)
        _ = _remove_cast(ori_reduce_funcs[1], dtype_stack_b)

        max_depth_ab = 0
        for d in range(min(len(dtype_stack_a), len(dtype_stack_b))):
            if dtype_stack_a[d] == dtype_stack_b[d]:
                max_depth_ab = d + 1
            else:
                break

        # FIXME 上述产生max_depth_ab的算法和注释中的说明并不是一致的,上述算法max_depth_ab==1的条件主要取决于b矩阵是否是kernel的一个输入
        # 若是的话,那么b大概率在tvm的转换中也只可能带上一组T.Cast
        # 若b也是由某些计算得到的,那么就不一定了,出现这种case的时候再修改吧...
        assert max_depth_ab == 1

        # 对a和b同时调用max_depth_ab次,那么就能得到对应的sub_reduce_func,再重复非GEMM的情况产生input buffer即可
        sub_reduce_funcs = list(map(lambda ori_reduce_func: reduce(lambda expr, _: expr.value, range(max_depth_ab), ori_reduce_func), ori_reduce_funcs))
    else:
        # 非GEMM op应该啥也不用干,直接产生input buffer,同时做替换
        sub_reduce_funcs = [ori_reduce_func]

    # BAD NAME!
    replaced_reduce_func = ori_reduce_func
    for sub_idx, sub_reduce_func in enumerate(sub_reduce_funcs):
        sub_reduce_func_loads = []
        tir.stmt_functor.post_order_visit(sub_reduce_func, _collect_sub_func_loads)

        # 找到表达式中真实的输入(即x变量)
        needed_reduce_func_load = None
        for load in sub_reduce_func_loads:
            if reduction_infos.var_map[str(load)].startswith("x"):
                needed_reduce_func_load = load
                break

        # 如果sub_reduce_func本身就是一个BufferLoad的话,那么就不需要构造新的input了
        if not isinstance(sub_reduce_func, tir.BufferLoad):
            input_buffer = tir.decl_buffer(
                shape=needed_reduce_func_load.buffer.shape,
                dtype=sub_reduce_func.dtype, # Attention here
                name=f"input_{idx}_{sub_idx}"
            )

            input_buffer_load = tir.BufferLoad(
                buffer=input_buffer,
                indices=_indices_to_indices(needed_reduce_func_load.indices)
            )

            # _replace_sub函数顶格就会判断Cast,所以对于GEMM的情况,这里直接传replaced_reduce_func是有问题的
            # 以前写的是对的,就是这样可能有点误解,对于GEMM的左右两边,都尝试对两边进行替换了,实际上都只需要一边就好了
            if block_info.is_gemm():
                assert isinstance(replaced_reduce_func, tir.Mul)
                replaced_reduce_func = tir.Mul(
                    a=_replace_sub(replaced_reduce_func.a, sub_reduce_func, input_buffer_load, max_depth_ab),
                    b=_replace_sub(replaced_reduce_func.b, sub_reduce_func, input_buffer_load, max_depth_ab)
                )
            else:
                replaced_reduce_func = _replace_sub(replaced_reduce_func, sub_reduce_func, input_buffer_load, 0) # 这里depth参数填0的话,遇到T.Cast也会跳过的

            input_buffer_stmt = tir.BufferStore(
                buffer=input_buffer,
                value=sub_reduce_func,
                indices=_indices_to_indices(needed_reduce_func_load.indices)
            )

            input_buffers.append(input_buffer)
            input_buffer_stmts.append(input_buffer_stmt)

    # 2.3 do_rescale(if needed)
    if need_cal_rescale_factor:
        rescale_factor_buffer_load = tir.BufferLoad(
            buffer=rescale_factor_buffer,
            indices=_indices_to_indices(ew_buffer_load.indices)
        )

        do_rescale_value = tir.Mul(reduce_target_buffer_load, rescale_factor_buffer_load)
        do_rescale_stmt = tir.BufferStore(
            buffer=reduce_target_buffer,
            value=do_rescale_value,
            indices=_indices_to_indices(reduce_target_buffer_load.indices)
        )
    else:
        do_rescale_stmt = None

    # 2.4 do_reduce
    # y(reduce_target) = y reduce_op replaced_reduce_func
    # if GEMM: y = y + replaced_reduce_func
    if reduction_config.reduce_op == "+":
        do_reduce_value = tir.Add(
            a=tir.BufferLoad(reduce_target_buffer, _indices_to_indices(reduce_target_buffer_load.indices)),
            b=replaced_reduce_func
        )
    elif reduction_config.reduce_op == "max":
        do_reduce_value = tir.Max(
            a=tir.BufferLoad(reduce_target_buffer, _indices_to_indices(reduce_target_buffer_load.indices)),
            b=replaced_reduce_func
        )
    do_reduce_stmt = tir.BufferStore(
        buffer=reduce_target_buffer,
        value=do_reduce_value,
        indices=_indices_to_indices(reduce_target_buffer_load.indices)
    )

    # 处理init,reads/writes相关
    # 对reduce_target进行初始化
    if reduction_config.reduce_op == "+":
        init_value = tir.FloatImm(reduce_target_buffer.dtype, 0.0)
    elif reduction_config.reduce_op == "max":
        # init_value = -tir.infinity(dtype=reduce_target_buffer.dtype)
        # 用一个较大的负数代替负无穷
        init_value = tir.FloatImm(reduce_target_buffer.dtype, -1e6)

    init = tir.BufferStore(
        buffer=reduce_target_buffer,
        value=init_value,
        indices=_indices_to_indices(reduce_target_buffer_load.indices)
    )

    # stmts
    stmts = []
    extra_allocates = []
    if need_update_prev_reduce:
        stmts.append(prev_reduce_stmt)
        extra_allocates.append(prev_reduce_buffer)
    if need_cal_rescale_factor:
        stmts.append(rescale_factor_stmt)
        extra_allocates.append(rescale_factor_buffer)
    stmts.append(*input_buffer_stmts)
    extra_allocates.append(*input_buffers)
    if need_cal_rescale_factor:
        stmts.append(do_rescale_stmt)
    stmts.append(do_reduce_stmt)

    # 产生待替换的block(没有reads/writes)
    tmp_block = tir.Block(
        iter_vars=block.iter_vars,
        reads=[],
        writes=[],
        name_hint=f"reduction{idx}",
        body=tir.SeqStmt(stmts),
        init=init
    )

    # reads/writes通过调用get_block_read_write_region得到,需要获取到该block中所有buffer的buffer_var_map
    buffer_var_map = {}

    def _collect_buffer_var_map(e):
        if isinstance(e, (tir.BufferLoad, tir.BufferStore)):
            buffer_var_map[e.buffer.data] = e.buffer

    # 这样来获取reads/writes就需要先把tmp_block生成出来才行,有没有更好的办法?
    # tmp_block.body应该一定是seqstmt(因为至少有input和do_reduce)
    for stmt in tmp_block.body:
        tir.stmt_functor.post_order_visit(stmt, _collect_buffer_var_map)

    reads_writes = tir.analysis.get_block_read_write_region(tmp_block, buffer_var_map)

    # 产生最终的block(有reads/writes)
    new_block = tir.Block(
        iter_vars=tmp_block.iter_vars,
        reads=reads_writes[0],
        writes=reads_writes[1],
        name_hint=tmp_block.name_hint,
        body=tmp_block.body,
        init=tmp_block.init
    )

    # 在这里判断是否有prologue和epilogue
    # 保存可能的pro和ep的block name,然后在外面统一更换名字

    # 成为prologue的条件: producer不在当前cascaded_group中
    def _update_pros(block_rv):
        pro_idx = 0
        producers = sch.get_producers(block_rv)
        for producer in producers:
            producer_block = sch.get(producer)
            if producer_block in [sch.get(block_info.block_rv) for block_info in reduction_infos.cascaded_group]:
                continue

            # prologue
            if producer_block.name_hint not in pro_ep_map:
                pro_ep_map[producer_block.name_hint] = f"prologue{pro_idx}"
                pro_idx += 1

    # 成为epilogue的条件: consumer不在当前cascaded_group中
    def _update_eps(block_rv):
        ep_idx = 0
        consumers = sch.get_consumers(block_rv)
        for consumer in consumers:
            consumer_block = sch.get(consumer)
            if consumer_block in [sch.get(block_info.block_rv) for block_info in reduction_infos.cascaded_group]:
                continue

            # epilogue
            if consumer_block.name_hint not in pro_ep_map:
                pro_ep_map[consumer_block.name_hint] = f"epilogue{ep_idx}"
                ep_idx += 1

    _update_pros(block_rv)
    _update_eps(block_rv)

    return new_block, extra_allocates


def transform_reductions_with_split(
    reduction_infos: CascadedGroupInfo,
    reduce_funcs_list: list[tuple[BMat | None, BMat | None, BMat | None]],
    num_split: Optional[int]
):
    
    assert num_split is not None and num_split > 0

    assert len(reduction_infos.reduction_configs) == len(reduce_funcs_list)

    n = len(reduction_infos.reduction_configs)

    sch = reduction_infos.sch

    new_blocks = []
    extra_alloc_buffers = []

    pro_ep_map = {}

    # 循环每个要transform的block,得到新的block,额外分配的buffer并更新pro_ep_map
    # 在合适的位置插入split维度
    for idx in range(n):
        new_block, extra_allocates = transform_single_block_with_split(
            idx,
            reduction_infos,
            reduce_funcs_list,
            pro_ep_map,
            num_split
        )
        
        new_blocks.append(new_block)
        extra_alloc_buffers.extend(extra_allocates)

    # 循环替换new_block,并且为new_block的循环添加注释
    for idx in range(n):
        ori_block_info = reduction_infos.cascaded_group[idx]
        ori_block_rv = ori_block_info.block_rv
        ori_block = sch.get(ori_block_rv)
        ori_sref = sch.state.get_sref(ori_block)

        sch.state.replace(ori_sref, new_blocks[idx])

        # 找到规约变量并添加注释
        new_block_realize = _get_blockrealize(sch, sch.get_block(f"reduction{idx}"))
        reduce_idx = [idx for idx, iv in enumerate(sch.get(sch.get_block(f"reduction{idx}")).iter_vars) if iv.iter_type == 2][0]
        reduce_var = new_block_realize.iter_values[reduce_idx]

        new_loops = sch.get_loops(f"reduction{idx}")
        for loop_rv in new_loops:
            if sch.get(loop_rv).loop_var == reduce_var:
                sch.annotate(loop_rv, "tag", "fused")

            loop_name = sch.get(loop_rv).loop_var.name
            sch.annotate(loop_rv, "name", loop_name)

    # 替换prologue和epilogue block的名字,并更新注释
    for ori_name_hint, pro_ep_name_hint in pro_ep_map.items():
        block = sch.get(sch.get_block(ori_name_hint))
        block_ref = sch.state.get_sref(block)

        new_block = tir.Block(
            iter_vars=block.iter_vars,
            reads=block.reads,
            writes=block.writes,
            name_hint=pro_ep_name_hint,
            body=block.body,
            init=block.init,
            alloc_buffers=block.alloc_buffers,
            match_buffers=block.match_buffers,
            annotations=block.annotations
        )

        sch.state.replace(block_ref, new_block)

        # reduce_var在这里复用了,理论上上面的循环无论哪个都应该得到的是相同的var
        # 另外这里直接判断var是不行的,应该是因为在跨block比较变量,所以只能判断名字是不是相同了
        new_loops = sch.get_loops(pro_ep_name_hint)
        for loop_rv in new_loops:
            loop_name = sch.get(loop_rv).loop_var.name
            if loop_name == reduce_var.name:
                sch.annotate(loop_rv, "tag", "fused")
            sch.annotate(loop_rv, "name", loop_name)

    # 添加bind annotations
    # 实现方式: 将所有的reduction,prologue,epilogue中的loop一起进行遍历,找到第一个不完全相同的位置,将前面完全相同的部分进行bind
    bind_block_name_hints = [f"reduction{idx}" for idx in range(n)]
    bind_block_name_hints.extend(list(pro_ep_map.values()))

    t = 0
    max_t = min([len(sch.get_loops(bind_block_name_hint)) for bind_block_name_hint in bind_block_name_hints])
    while t < max_t:
        loop_names = list(set([sch.get(sch.get_loops(bind_block_name_hint)[t]).loop_var.name for bind_block_name_hint in bind_block_name_hints]))
        if len(loop_names) > 1:
            break
        if loop_names[0] != reduce_var.name:
            t = t + 1
        else:
            break

    for idx in range(t):
        loop_rvs = [sch.get_loops(bind_block_name_hint)[idx] for bind_block_name_hint in bind_block_name_hints]
        for loop_rv in loop_rvs:
            bind_idx = f"vblockIdx.{t - idx - 1}"
            sch.annotate(loop_rv, "bind", bind_idx)

    # 得到root block,用来更新新分配的alloc_buffers
    new_block0_sref = sch.state.get_sref(new_blocks[0])
    parent_sref = new_block0_sref.parent
    while parent_sref is not None:
        stmt = parent_sref.stmt
        if isinstance(stmt, tir.Block):
            root_block = stmt
            break
        parent_sref = parent_sref.parent

    assert root_block is not None

    root_block_sref = sch.state.get_sref(root_block)

    all_alloc_buffers = list(root_block.alloc_buffers)
    all_alloc_buffers.extend(extra_alloc_buffers)

    new_root_block = tir.Block(
        iter_vars=root_block.iter_vars,
        reads=root_block.reads,
        writes=root_block.writes,
        name_hint=root_block.name_hint,
        body=root_block.body,
        init=root_block.init,
        alloc_buffers=all_alloc_buffers,
        match_buffers=root_block.match_buffers,
        annotations=root_block.annotations
    )

    sch.state.replace(root_block_sref, new_root_block)
