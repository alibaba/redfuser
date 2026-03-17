"""
TIR Online Fusion Generator (Split Version)

Transforms cascaded reduction Blocks into Online algorithm form with split processing.
Each Block contains 4 steps:
1. Update prev_reduce
2. Calculate rescale_factor and reduction input
3. do_rescale
4. do_reduce

The Split version divides the reduction axis into num_split segments, generating Kernel1:
- Kernel1: Processes reduction for each split segment, outputs partial results
- Kernel2: Merges partial results (implemented by other modules)

Key differences from non-split version:
1. New splitp dimension: All reduction blocks' iter_vars include a Spatial type v_splitp
2. Buffer shape transformation:
   - External buffers (function params): Shape unchanged, accessed via compound index
   - Internal y-type buffers (reduction target, no reduce axis): Append num_split at end
   - Internal x-type buffers (with reduce axis): Insert num_split before reduce axis, reduce extent becomes extent/num_split
3. Index transformation:
   - External buffer + has reduce var: reduce_var -> splitp * chunk_size + reduce_var
   - Internal buffer + has reduce var: Insert splitp before reduce_var
   - Internal buffer + no reduce var: Append splitp at end
4. Loop structure: Insert splitp loop between outer spatial loops and reduction loop
5. Epilogue: Write out partial results to output buffer
"""
from typing import Optional, List, Set, Tuple, Dict, Any

import tvm
from tvm import tir

from functools import reduce
import sympy as sp

from .tir_reduction_analyzer import CascadedGroupInfo
from .utils import BMat, ElementwiseApplyNAry


def _create_split_buffer_shape(
    original_shape: List[tir.PrimExpr],
    num_split: int,
    is_y_type: bool,
    reduce_axis_pos: Optional[int] = None
) -> List[tir.PrimExpr]:
    """
    Create split version of buffer shape based on buffer type.

    Args:
        original_shape: Original buffer shape
        num_split: Number of splits
        is_y_type: True for y-type buffer (no reduce axis), False for x-type buffer (has reduce axis)
        reduce_axis_pos: Position of reduce axis in x-type buffer

    Returns:
        Split version of shape
    """
    shape_list = list(original_shape)

    if is_y_type:
        # y-type: Append num_split dimension at end
        shape_list.append(num_split)
    else:
        # x-type: Insert num_split before reduce axis, reduce extent becomes extent/num_split
        assert reduce_axis_pos is not None
        original_extent = shape_list[reduce_axis_pos]
        chunk_size = original_extent // num_split
        shape_list[reduce_axis_pos] = chunk_size
        shape_list.insert(reduce_axis_pos, num_split)

    return shape_list


def transform_single_block_with_split(
    idx: int,
    reduction_infos: CascadedGroupInfo,
    reduce_funcs_list: List[Tuple[BMat | None, BMat | None, BMat | None]],
    pro_ep_map: dict,
    num_split: int,
    func_param_buffers: Set[tir.Buffer],
    reduce_var_name: str,
    reduce_extent: int,
    buffer_remap: Dict[tir.Buffer, tir.Buffer],
    chunk_size: int,
    splitp_var: tir.Var,
):
    """
    Generate split-version online algorithm block for a single reduction block.

    Key differences from non-split version:
    1. Add splitp iter_var
    2. Modify buffer access indices (internal buffer vs external buffer)
    3. Use buffer_remap to get split version of buffer

    Args:
        idx: Block index
        reduction_infos: Cascaded reduction group info
        reduce_funcs_list: List of reduction functions
        pro_ep_map: Prologue/epilogue mapping
        num_split: Number of splits
        func_param_buffers: Set of function parameter buffers (for distinguishing internal/external buffers)
        reduce_var_name: Reduction variable name
        reduce_extent: Original reduction axis extent
        buffer_remap: Mapping from original buffer to split version buffer
        chunk_size: Reduction length per split
        splitp_var: splitp variable

    Returns:
        new_block: New block
        extra_allocates: List of extra allocated buffers
    """
    assert num_split is not None and num_split > 0

    sch = reduction_infos.sch
    block_info = reduction_infos.cascaded_group[idx]
    block_rv = block_info.block_rv
    block = sch.get(block_rv)
    reduction_config = reduction_infos.reduction_configs[idx]

    # Create independent splitp variable for each block (not shared, to avoid TIR well-formedness issues)
    block_splitp_var = tir.Var("v_splitp", "int32")

    # Important: Create independent tir.Var copies for all iter_vars of each block
    # This must be done before body generation to ensure body uses independent variables
    block_iter_vars_copies = {}  # Store independent copies for later new_iter_vars construction
    iter_vars_map = {}
    for iv in block.iter_vars:
        # Create independent tir.Var copy
        new_var = tir.Var(iv.var.name, iv.var.dtype)
        block_iter_vars_copies[iv.var.name] = (new_var, iv)  # (new variable, original iter_var)
        iter_vars_map[iv.var.name] = new_var
    iter_vars_map["v_splitp"] = block_splitp_var

    # Find reduce variable
    reduce_var = None
    for iv in block.iter_vars:
        if iv.iter_type == 2:  # CommReduce
            reduce_var = iv.var
            break
    assert reduce_var is not None

    def _is_external_buffer(buffer: tir.Buffer) -> bool:
        """Check if buffer is external buffer (function parameter)"""
        return buffer in func_param_buffers

    def _indices_contain_reduce_var(indices: List[tir.PrimExpr]) -> bool:
        """Check if indices contain reduce variable"""
        for idx_expr in indices:
            if isinstance(idx_expr, tir.Var) and idx_expr.name == reduce_var_name:
                return True
        return False

    def _get_remapped_buffer(buffer: tir.Buffer) -> tir.Buffer:
        """Get remapped buffer, external buffer is not remapped"""
        if _is_external_buffer(buffer):
            return buffer
        return buffer_remap.get(buffer, buffer)

    def _split_indices_to_indices(
        indices: List[tir.PrimExpr],
        buffer: tir.Buffer
    ) -> Tuple[tir.Buffer, List[tir.PrimExpr]]:
        """
        Convert original indices to split version indices and return remapped buffer.

        Rules:
        - External buffer + has reduce var: Replace reduce_var with block_splitp_var * chunk_size + reduce_var
        - Internal buffer + has reduce var: Insert block_splitp_var before reduce_var
        - Internal buffer + no reduce var: Append block_splitp_var at end
        """
        is_external = _is_external_buffer(buffer)
        contains_reduce = _indices_contain_reduce_var(indices)
        remapped_buffer = _get_remapped_buffer(buffer)

        out_indices = []
        for v in indices:
            if isinstance(v, tir.Var):
                if v.name == reduce_var_name:
                    if is_external:
                        # External buffer: Use compound index
                        out_indices.append(
                            block_splitp_var * chunk_size + iter_vars_map.get(v.name, v)
                        )
                    else:
                        # Internal buffer: Insert splitp first, then reduce_var
                        out_indices.append(block_splitp_var)
                        out_indices.append(iter_vars_map.get(v.name, v))
                else:
                    out_indices.append(iter_vars_map.get(v.name, v))
            else:
                out_indices.append(v)

        # Internal buffer + no reduce var: Append block_splitp_var at end
        if not is_external and not contains_reduce:
            out_indices.append(block_splitp_var)

        return remapped_buffer, out_indices

    def _indices_to_indices(indices: List[tir.Var]) -> List[tir.PrimExpr]:
        """Simple index mapping (for cases not requiring split transformation)"""
        name_indices = [v.name for v in indices]
        out_indices = list(map(lambda name: iter_vars_map[name], name_indices))
        return out_indices

    # 1. Check if prev_reduce needs to be updated
    def _need_update_prev_reduce(block_rv):
        consumers = sch.get_consumers(block_rv)
        for consumer in consumers:
            if sch.get(consumer) in [sch.get(bi.block_rv) for bi in reduction_infos.cascaded_group]:
                return True
        return False

    # Get reduce_target related info
    reduce_target_buffer_load = reduction_infos.y_buffer_load_map[reduction_config.reduce_target]
    reduce_target_buffer = reduce_target_buffer_load.buffer
    # Get split version of reduce_target_buffer
    split_reduce_target_buffer = buffer_remap.get(reduce_target_buffer, reduce_target_buffer)

    need_update_prev_reduce = _need_update_prev_reduce(block_rv)
    if need_update_prev_reduce:
        # prev_reduce_buffer shape should match split version of reduce_target_buffer
        prev_reduce_buffer = tir.decl_buffer(
            shape=split_reduce_target_buffer.shape,
            dtype=split_reduce_target_buffer.dtype,
            name=f"prev_{reduce_target_buffer.name}"
        )

        # Use split version indices
        _, prev_indices = _split_indices_to_indices(
            reduce_target_buffer_load.indices, reduce_target_buffer
        )

        prev_reduce_stmt = tir.BufferStore(
            buffer=prev_reduce_buffer,
            value=tir.BufferLoad(split_reduce_target_buffer, prev_indices),
            indices=prev_indices
        )

        # Update prev_buffer_load_map - Note: Create a "fake" BufferLoad for subsequent sympy_expr_to_tir
        # We create a temp buffer (original shape) to store indices info, actual buffer will be replaced in _split_indices_to_indices
        # Because sympy_expr_to_tir will call _split_indices_to_indices to transform indices
        # So we need to keep original indices but point to prev_reduce_buffer
        # Trick: Add a mapping from temp buffer to prev_reduce_buffer in buffer_remap
        temp_prev_buffer = tir.decl_buffer(
            shape=reduce_target_buffer.shape,  # Original shape to maintain consistent indices length
            dtype=reduce_target_buffer.dtype,
            name=f"prev_{reduce_target_buffer.name}_temp"
        )
        buffer_remap[temp_prev_buffer] = prev_reduce_buffer  # Add mapping
        reduction_infos.prev_buffer_load_map[f"prev_{reduction_config.reduce_target}"] = \
            tvm.tir.BufferLoad(buffer=temp_prev_buffer, indices=reduce_target_buffer_load.indices)
    else:
        prev_reduce_buffer = None
        prev_reduce_stmt = None

    # 2.1 Check if rescale_factor needs to be calculated
    def _need_cal_rescale_factor(block_rv):
        producers = sch.get_producers(block_rv)
        for producer in producers:
            if sch.get(producer) in [sch.get(bi.block_rv) for bi in reduction_infos.cascaded_group]:
                return True
        return False

    def sympy_expr_to_tir(expr, ew_map: dict):
        """Convert sympy expression to TIR expression"""
        if isinstance(expr, BMat):
            return sympy_expr_to_tir(expr.expr, ew_map)

        if isinstance(expr, ElementwiseApplyNAry):
            variables = expr.function.variables
            operands = expr.operands
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
            elif xyc.startswith("c"):
                buffer_load = reduction_infos.c_buffer_load_map[xyc]
            elif xyc.startswith("prev"):
                buffer_load = reduction_infos.prev_buffer_load_map[xyc]
            else:
                buffer_load = None

            # Use split version indices and buffer
            remapped_buf, split_indices = _split_indices_to_indices(buffer_load.indices, buffer_load.buffer)
            return tir.BufferLoad(remapped_buf, split_indices)

        if expr.is_Number:
            return tir.FloatImm("float32", float(expr))

        raise NotImplementedError(f"Unknown expr type: {type(expr)}")

    need_cal_rescale_factor = _need_cal_rescale_factor(block_rv)
    if need_cal_rescale_factor:
        H_expr = reduce_funcs_list[idx][1]
        prev_H_expr = reduce_funcs_list[idx][2]
        rescale_factor_expr = H_expr / prev_H_expr

        ew_map = {}
        rescale_factor_value = sympy_expr_to_tir(rescale_factor_expr, ew_map)

        ew_buffer_load = reduction_infos.y_buffer_load_map[list(ew_map.values())[0]] \
            if list(ew_map.values())[0].startswith('y') \
            else reduction_infos.prev_buffer_load_map[list(ew_map.values())[0]]

        # rescale_factor_buffer shape should be based on split y-type buffer
        split_ew_buffer = _get_remapped_buffer(ew_buffer_load.buffer)
        rescale_factor_buffer = tir.decl_buffer(
            shape=split_ew_buffer.shape,
            dtype=split_ew_buffer.dtype,
            name=f"rescale_factor_{idx}"
        )

        _, rescale_indices = _split_indices_to_indices(ew_buffer_load.indices, ew_buffer_load.buffer)

        rescale_factor_stmt = tir.BufferStore(
            buffer=rescale_factor_buffer,
            value=rescale_factor_value,
            indices=rescale_indices
        )
    else:
        rescale_factor_buffer = None
        rescale_factor_stmt = None
        ew_buffer_load = None

    # 2.2 Generate input data
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

    # Handle based on whether this is a GEMM
    if block_info.is_gemm():
        assert isinstance(ori_reduce_func, tir.Mul)
        ori_reduce_funcs = [ori_reduce_func.a, ori_reduce_func.b]

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

        assert max_depth_ab == 1
        sub_reduce_funcs = list(map(
            lambda ori_reduce_func: reduce(lambda expr, _: expr.value, range(max_depth_ab), ori_reduce_func),
            ori_reduce_funcs
        ))
    else:
        sub_reduce_funcs = [ori_reduce_func]

    replaced_reduce_func = ori_reduce_func
    for sub_idx, sub_reduce_func in enumerate(sub_reduce_funcs):
        sub_reduce_func_loads = []
        tir.stmt_functor.post_order_visit(sub_reduce_func, _collect_sub_func_loads)

        needed_reduce_func_load = None
        for load in sub_reduce_func_loads:
            if reduction_infos.var_map[str(load)].startswith("x"):
                needed_reduce_func_load = load
                break

        if not isinstance(sub_reduce_func, tir.BufferLoad):
            # input buffer shape should be based on split x-type buffer
            split_needed_buffer = _get_remapped_buffer(needed_reduce_func_load.buffer)
            input_buffer = tir.decl_buffer(
                shape=split_needed_buffer.shape,
                dtype=sub_reduce_func.dtype,
                name=f"input_{idx}_{sub_idx}"
            )

            # Use split version indices
            _, input_indices = _split_indices_to_indices(
                needed_reduce_func_load.indices, needed_reduce_func_load.buffer
            )

            input_buffer_load = tir.BufferLoad(
                buffer=input_buffer,
                indices=input_indices
            )

            if block_info.is_gemm():
                assert isinstance(replaced_reduce_func, tir.Mul)
                replaced_reduce_func = tir.Mul(
                    a=_replace_sub(replaced_reduce_func.a, sub_reduce_func, input_buffer_load, max_depth_ab),
                    b=_replace_sub(replaced_reduce_func.b, sub_reduce_func, input_buffer_load, max_depth_ab)
                )
            else:
                replaced_reduce_func = _replace_sub(replaced_reduce_func, sub_reduce_func, input_buffer_load, 0)

            # Construct input buffer assignment statement, need to transform BufferLoad indices in sub_reduce_func
            def _transform_sub_reduce_func(expr):
                """Recursively transform BufferLoad indices in sub_reduce_func"""
                if isinstance(expr, tir.BufferLoad):
                    remapped_buf, new_indices = _split_indices_to_indices(expr.indices, expr.buffer)
                    return tir.BufferLoad(remapped_buf, new_indices)
                elif isinstance(expr, tir.Cast):
                    return tir.Cast(expr.dtype, _transform_sub_reduce_func(expr.value))
                elif isinstance(expr, tir.Add):
                    return tir.Add(_transform_sub_reduce_func(expr.a), _transform_sub_reduce_func(expr.b))
                elif isinstance(expr, tir.Sub):
                    return tir.Sub(_transform_sub_reduce_func(expr.a), _transform_sub_reduce_func(expr.b))
                elif isinstance(expr, tir.Mul):
                    return tir.Mul(_transform_sub_reduce_func(expr.a), _transform_sub_reduce_func(expr.b))
                elif isinstance(expr, tir.Div):
                    return tir.Div(_transform_sub_reduce_func(expr.a), _transform_sub_reduce_func(expr.b))
                elif isinstance(expr, tir.Call):
                    new_args = [_transform_sub_reduce_func(arg) for arg in expr.args]
                    return tir.Call(expr.dtype, expr.op, new_args)
                else:
                    return expr

            transformed_sub_reduce_func = _transform_sub_reduce_func(sub_reduce_func)

            input_buffer_stmt = tir.BufferStore(
                buffer=input_buffer,
                value=transformed_sub_reduce_func,
                indices=input_indices
            )

            input_buffers.append(input_buffer)
            input_buffer_stmts.append(input_buffer_stmt)
        else:
            # When sub_reduce_func is already a BufferLoad (e.g., V buffer),
            # still need to apply split index transformation
            remapped_buf, new_indices = _split_indices_to_indices(
                sub_reduce_func.indices, sub_reduce_func.buffer
            )
            transformed_load = tir.BufferLoad(remapped_buf, new_indices)
            # Update corresponding BufferLoad in replaced_reduce_func
            if block_info.is_gemm():
                assert isinstance(replaced_reduce_func, tir.Mul)
                replaced_reduce_func = tir.Mul(
                    a=_replace_sub(replaced_reduce_func.a, sub_reduce_func, transformed_load, max_depth_ab),
                    b=_replace_sub(replaced_reduce_func.b, sub_reduce_func, transformed_load, max_depth_ab)
                )
            else:
                replaced_reduce_func = _replace_sub(replaced_reduce_func, sub_reduce_func, transformed_load, 0)

    # 2.3 do_rescale(if needed)
    if need_cal_rescale_factor:
        _, rescale_indices = _split_indices_to_indices(ew_buffer_load.indices, ew_buffer_load.buffer)
        rescale_factor_buffer_load = tir.BufferLoad(
            buffer=rescale_factor_buffer,
            indices=rescale_indices
        )

        _, reduce_target_split_indices = _split_indices_to_indices(
            reduce_target_buffer_load.indices, reduce_target_buffer
        )

        do_rescale_value = tir.Mul(
            tir.BufferLoad(split_reduce_target_buffer, reduce_target_split_indices),
            rescale_factor_buffer_load
        )
        do_rescale_stmt = tir.BufferStore(
            buffer=split_reduce_target_buffer,
            value=do_rescale_value,
            indices=reduce_target_split_indices
        )
    else:
        do_rescale_stmt = None

    # 2.4 do_reduce
    _, reduce_target_split_indices = _split_indices_to_indices(
        reduce_target_buffer_load.indices, reduce_target_buffer
    )

    if reduction_config.reduce_op == "+":
        do_reduce_value = tir.Add(
            a=tir.BufferLoad(split_reduce_target_buffer, reduce_target_split_indices),
            b=replaced_reduce_func
        )
    elif reduction_config.reduce_op == "max":
        do_reduce_value = tir.Max(
            a=tir.BufferLoad(split_reduce_target_buffer, reduce_target_split_indices),
            b=replaced_reduce_func
        )
    else:
        raise NotImplementedError(f"Unsupported reduce_op in split version: {reduction_config.reduce_op}")

    do_reduce_stmt = tir.BufferStore(
        buffer=split_reduce_target_buffer,
        value=do_reduce_value,
        indices=reduce_target_split_indices
    )

    # Handle init
    if reduction_config.reduce_op == "+":
        init_value = tir.FloatImm(split_reduce_target_buffer.dtype, 0.0)
    elif reduction_config.reduce_op == "max":
        init_value = tir.FloatImm(split_reduce_target_buffer.dtype, -1e6)

    init = tir.BufferStore(
        buffer=split_reduce_target_buffer,
        value=init_value,
        indices=reduce_target_split_indices
    )

    # Construct statement sequence
    stmts = []
    extra_allocates = []
    if need_update_prev_reduce:
        stmts.append(prev_reduce_stmt)
        extra_allocates.append(prev_reduce_buffer)
    if need_cal_rescale_factor:
        stmts.append(rescale_factor_stmt)
        extra_allocates.append(rescale_factor_buffer)
    stmts.extend(input_buffer_stmts)
    extra_allocates.extend(input_buffers)
    if need_cal_rescale_factor:
        stmts.append(do_rescale_stmt)
    stmts.append(do_reduce_stmt)

    # Construct new iter_vars, insert splitp before reduce axis
    # Also update reduce axis extent to chunk_size
    # Use independent variable copies created at function start (stored in block_iter_vars_copies)
    new_iter_vars = []
    reduce_pos = None
    for i, iv in enumerate(block.iter_vars):
        var_name = iv.var.name
        new_var, orig_iv = block_iter_vars_copies[var_name]
        
        if iv.iter_type == 2 and reduce_pos is None:  # CommReduce
            reduce_pos = i
            # Insert splitp iter_var - use block_splitp_var created at function start
            split_iter_var = tir.IterVar(
                dom=tvm.ir.Range(0, num_split),
                var=block_splitp_var,
                iter_type=0,  # DataPar (Spatial)
                thread_tag=""
            )
            new_iter_vars.append(split_iter_var)
            # Modify reduce iter_var extent to chunk_size
            # Use pre-created independent variable copy
            new_reduce_iter_var = tir.IterVar(
                dom=tvm.ir.Range(0, chunk_size),
                var=new_var,  # Use pre-created independent variable
                iter_type=iv.iter_type,
                thread_tag=iv.thread_tag
            )
            new_iter_vars.append(new_reduce_iter_var)
        else:
            # Spatial iter_var uses pre-created independent variable copy
            new_spatial_iter_var = tir.IterVar(
                dom=iv.dom,
                var=new_var,  # Use pre-created independent variable
                iter_type=iv.iter_type,
                thread_tag=iv.thread_tag
            )
            new_iter_vars.append(new_spatial_iter_var)

    # Handle body
    if len(stmts) == 1:
        body = stmts[0]
    else:
        body = tir.SeqStmt(stmts)

    # Create temporary block to be replaced (without reads/writes)
    tmp_block = tir.Block(
        iter_vars=new_iter_vars,
        reads=[],
        writes=[],
        name_hint=f"reduction{idx}",
        body=body,
        init=init
    )

    # Collect buffer_var_map for computing reads/writes
    buffer_var_map = {}

    def _collect_buffer_var_map(e):
        if isinstance(e, (tir.BufferLoad, tir.BufferStore)):
            buffer_var_map[e.buffer.data] = e.buffer

    if isinstance(tmp_block.body, tir.SeqStmt):
        for stmt in tmp_block.body:
            tir.stmt_functor.post_order_visit(stmt, _collect_buffer_var_map)
    else:
        tir.stmt_functor.post_order_visit(tmp_block.body, _collect_buffer_var_map)

    # Also collect from init
    if tmp_block.init is not None:
        tir.stmt_functor.post_order_visit(tmp_block.init, _collect_buffer_var_map)

    reads_writes = tir.analysis.get_block_read_write_region(tmp_block, buffer_var_map)

    # Create final block (with reads/writes)
    new_block = tir.Block(
        iter_vars=tmp_block.iter_vars,
        reads=reads_writes[0],
        writes=reads_writes[1],
        name_hint=tmp_block.name_hint,
        body=tmp_block.body,
        init=tmp_block.init
    )

    # Update pro_ep_map
    def _update_pros(block_rv):
        pro_idx = len([k for k in pro_ep_map.values() if k.startswith("prologue")])
        producers = sch.get_producers(block_rv)
        for producer in producers:
            producer_block = sch.get(producer)
            if producer_block in [sch.get(bi.block_rv) for bi in reduction_infos.cascaded_group]:
                continue
            if producer_block.name_hint not in pro_ep_map:
                pro_ep_map[producer_block.name_hint] = f"prologue{pro_idx}"
                pro_idx += 1

    _update_pros(block_rv)

    return new_block, extra_allocates


def transform_prologue_with_split(
    prologue_block: tir.Block,
    num_split: int,
    func_param_buffers: Set[tir.Buffer],
    reduce_var_name: str,
    buffer_remap: Dict[tir.Buffer, tir.Buffer],
    chunk_size: int,
    splitp_var: tir.Var,
    new_name: str,
) -> tir.Block:
    """
    Transform prologue block to split version.

    Args:
        prologue_block: Original prologue block
        num_split: Number of splits
        func_param_buffers: Set of function parameter buffers
        reduce_var_name: Name of reduction variable
        buffer_remap: Buffer mapping
        chunk_size: Reduction length per split
        splitp_var: splitp variable
        new_name: New block name

    Returns:
        Transformed prologue block
    """
    # Create independent splitp variable for prologue block
    prologue_splitp_var = tir.Var("v_splitp", "int32")
    
    # Important: Create independent tir.Var copies for each iter_var
    # This must be done before body generation to ensure body uses independent variables
    prologue_iter_vars_copies = {}  # Store independent copies
    iter_vars_map = {}
    for iv in prologue_block.iter_vars:
        new_var = tir.Var(iv.var.name, iv.var.dtype)
        prologue_iter_vars_copies[iv.var.name] = (new_var, iv)
        iter_vars_map[iv.var.name] = new_var
    iter_vars_map["v_splitp"] = prologue_splitp_var

    def _is_external_buffer(buffer: tir.Buffer) -> bool:
        return buffer in func_param_buffers

    def _indices_contain_reduce_var(indices: List[tir.PrimExpr]) -> bool:
        for idx_expr in indices:
            if isinstance(idx_expr, tir.Var) and idx_expr.name == reduce_var_name:
                return True
        return False

    def _get_remapped_buffer(buffer: tir.Buffer) -> tir.Buffer:
        if _is_external_buffer(buffer):
            return buffer
        return buffer_remap.get(buffer, buffer)

    def _transform_indices(indices: List[tir.PrimExpr], buffer: tir.Buffer) -> Tuple[tir.Buffer, List[tir.PrimExpr]]:
        """Transform indices and return remapped buffer"""
        is_external = _is_external_buffer(buffer)
        contains_reduce = _indices_contain_reduce_var(indices)
        remapped_buffer = _get_remapped_buffer(buffer)

        out_indices = []
        for v in indices:
            if isinstance(v, tir.Var):
                if v.name == reduce_var_name:
                    if is_external:
                        out_indices.append(prologue_splitp_var * chunk_size + iter_vars_map.get(v.name, v))
                    else:
                        out_indices.append(prologue_splitp_var)
                        out_indices.append(iter_vars_map.get(v.name, v))
                else:
                    out_indices.append(iter_vars_map.get(v.name, v))
            else:
                out_indices.append(v)

        if not is_external and not contains_reduce:
            out_indices.append(prologue_splitp_var)

        return remapped_buffer, out_indices

    def _transform_expr(expr):
        """Recursively transform BufferLoad in expression"""
        if isinstance(expr, tir.BufferLoad):
            remapped_buf, new_indices = _transform_indices(list(expr.indices), expr.buffer)
            return tir.BufferLoad(remapped_buf, new_indices)
        elif isinstance(expr, tir.Cast):
            return tir.Cast(expr.dtype, _transform_expr(expr.value))
        elif isinstance(expr, tir.Add):
            return tir.Add(_transform_expr(expr.a), _transform_expr(expr.b))
        elif isinstance(expr, tir.Sub):
            return tir.Sub(_transform_expr(expr.a), _transform_expr(expr.b))
        elif isinstance(expr, tir.Mul):
            return tir.Mul(_transform_expr(expr.a), _transform_expr(expr.b))
        elif isinstance(expr, tir.Div):
            return tir.Div(_transform_expr(expr.a), _transform_expr(expr.b))
        elif isinstance(expr, tir.Call):
            new_args = [_transform_expr(arg) for arg in expr.args]
            return tir.Call(expr.dtype, expr.op, new_args)
        elif isinstance(expr, (tir.IntImm, tir.FloatImm)):
            return expr
        elif isinstance(expr, tir.Var):
            return iter_vars_map.get(expr.name, expr)
        else:
            return expr

    def _transform_stmt(stmt):
        """Recursively transform statement"""
        if isinstance(stmt, tir.BufferStore):
            remapped_buf, new_indices = _transform_indices(list(stmt.indices), stmt.buffer)
            new_value = _transform_expr(stmt.value)
            return tir.BufferStore(remapped_buf, new_value, new_indices)
        elif isinstance(stmt, tir.SeqStmt):
            return tir.SeqStmt([_transform_stmt(s) for s in stmt])
        else:
            return stmt

    # Transform body
    new_body = _transform_stmt(prologue_block.body)

    # Transform init (if present)
    new_init = None
    if prologue_block.init is not None:
        new_init = _transform_stmt(prologue_block.init)

    # Construct new iter_vars, insert splitp before kv_len axis
    # Note: For prologue (QK matmul), kv_len is Spatial type (output dimension)
    # while head_dim_qk is CommReduce type (reduction dimension)
    # So must search by variable name, not by iter_type
    # Use pre-created independent variable copies (stored in prologue_iter_vars_copies)
    new_iter_vars = []
    for iv in prologue_block.iter_vars:
        var_name = iv.var.name
        new_var, orig_iv = prologue_iter_vars_copies[var_name]
        
        if var_name == reduce_var_name:  # Search for kv_len axis by name
            # Insert splitp (Spatial) before kv_len - use prologue_splitp_var created at function start
            split_iter_var = tir.IterVar(
                dom=tvm.ir.Range(0, num_split),
                var=prologue_splitp_var,
                iter_type=0,  # Spatial
                thread_tag=""
            )
            new_iter_vars.append(split_iter_var)
            # Change kv_len extent to chunk_size, use pre-created independent variable
            new_iter_var = tir.IterVar(
                dom=tvm.ir.Range(0, chunk_size),
                var=new_var,  # Use pre-created independent variable
                iter_type=iv.iter_type,  # Keep original type
                thread_tag=iv.thread_tag
            )
            new_iter_vars.append(new_iter_var)
        else:
            # Use pre-created independent variable copy
            new_iter_var = tir.IterVar(
                dom=iv.dom,
                var=new_var,  # Use pre-created independent variable
                iter_type=iv.iter_type,
                thread_tag=iv.thread_tag
            )
            new_iter_vars.append(new_iter_var)

    # Create temporary block for computing reads/writes
    tmp_block = tir.Block(
        iter_vars=new_iter_vars,
        reads=[],
        writes=[],
        name_hint=new_name,
        body=new_body,
        init=new_init
    )

    # Collect buffer_var_map
    buffer_var_map = {}

    def _collect_buffer_var_map(e):
        if isinstance(e, (tir.BufferLoad, tir.BufferStore)):
            buffer_var_map[e.buffer.data] = e.buffer

    tir.stmt_functor.post_order_visit(tmp_block.body, _collect_buffer_var_map)
    if tmp_block.init is not None:
        tir.stmt_functor.post_order_visit(tmp_block.init, _collect_buffer_var_map)

    reads_writes = tir.analysis.get_block_read_write_region(tmp_block, buffer_var_map)

    new_block = tir.Block(
        iter_vars=new_iter_vars,
        reads=reads_writes[0],
        writes=reads_writes[1],
        name_hint=new_name,
        body=new_body,
        init=new_init
    )

    return new_block


def build_epilogue_block(
    reduction_infos: CascadedGroupInfo,
    buffer_remap: Dict[tir.Buffer, tir.Buffer],
    part_output_buffers: Dict[str, tir.Buffer],
    num_split: int,
    splitp_var: tir.Var,
    common_spatial_vars: List[tir.Var],
    common_spatial_extents: List[int],
    extra_spatial_vars: List[tir.Var],
    extra_spatial_extents: List[int],
) -> tir.Block:
    """
    Construct epilogue block to write partial results to part_xxx output buffers.

    Args:
        reduction_infos: Cascaded reduction group info
        buffer_remap: Buffer mapping
        part_output_buffers: {reduce_target_name: part_xxx_buffer} mapping
        num_split: Number of splits
        splitp_var: splitp variable
        common_spatial_vars: Common spatial dimension variables
        common_spatial_extents: Common spatial dimension extents
        extra_spatial_vars: Extra spatial dimension variables (e.g., head_dim_v)
        extra_spatial_extents: Extra spatial dimension extents

    Returns:
        epilogue block
    """
    # Construct iter_vars: common_spatial + splitp + extra_spatial
    # Important: Create independent copies for each variable to avoid sharing with other blocks
    iter_vars = []
    epilogue_var_map = {}  # Store variable name to independent copy mapping, for body construction
    
    for var, extent in zip(common_spatial_vars, common_spatial_extents):
        # Create independent variable copy
        new_var = tir.Var(var.name, var.dtype)
        epilogue_var_map[var.name] = new_var
        iter_vars.append(tir.IterVar(
            dom=tvm.ir.Range(0, extent),
            var=new_var,  # Use independent variable
            iter_type=0,  # DataPar
            thread_tag=""
        ))

    # splitp - create new variable for epilogue block
    epilogue_splitp_var = tir.Var("v_splitp", "int32")
    epilogue_var_map["v_splitp"] = epilogue_splitp_var
    iter_vars.append(tir.IterVar(
        dom=tvm.ir.Range(0, num_split),
        var=epilogue_splitp_var,
        iter_type=0,
        thread_tag=""
    ))

    # extra spatial - also need to create independent variable copies
    for var, extent in zip(extra_spatial_vars, extra_spatial_extents):
        new_var = tir.Var(var.name, var.dtype)
        epilogue_var_map[var.name] = new_var
        iter_vars.append(tir.IterVar(
            dom=tvm.ir.Range(0, extent),
            var=new_var,  # Use independent variable
            iter_type=0,
            thread_tag=""
        ))

    # Construct body: Write out partial results for each reduction target
    stmts = []
    for reduction_config in reduction_infos.reduction_configs:
        reduce_target_name = reduction_config.reduce_target
        if reduce_target_name not in part_output_buffers:
            continue

        # Get original reduce_target buffer and split version
        reduce_target_buffer_load = reduction_infos.y_buffer_load_map[reduce_target_name]
        original_buffer = reduce_target_buffer_load.buffer
        split_buffer = buffer_remap.get(original_buffer, original_buffer)
        part_buffer = part_output_buffers[reduce_target_name]

        # Construct indices: common_spatial + splitp (+ extra_spatial if applicable)
        # Need to determine based on original buffer's indices structure
        original_indices = reduce_target_buffer_load.indices

        # split buffer indices: original indices' vars + splitp at end
        # Use epilogue_var_map to get independent variable copies
        split_indices = []
        for idx in original_indices:
            if isinstance(idx, tir.Var):
                # Use epilogue_var_map to find independent copy
                if idx.name in epilogue_var_map:
                    split_indices.append(epilogue_var_map[idx.name])
                else:
                    # If not found, may be external variable, keep as is
                    split_indices.append(idx)
            else:
                split_indices.append(idx)

        # Append splitp at end
        split_indices.append(epilogue_splitp_var)

        # part buffer indices same as split buffer indices
        part_indices = split_indices.copy()

        # Read split buffer and Cast to output dtype
        load_value = tir.BufferLoad(split_buffer, split_indices)
        if split_buffer.dtype != part_buffer.dtype:
            cast_value = tir.Cast(part_buffer.dtype, load_value)
        else:
            cast_value = load_value

        stmt = tir.BufferStore(part_buffer, cast_value, part_indices)
        stmts.append(stmt)

    if len(stmts) == 1:
        body = stmts[0]
    else:
        body = tir.SeqStmt(stmts)

    # Create temporary block
    tmp_block = tir.Block(
        iter_vars=iter_vars,
        reads=[],
        writes=[],
        name_hint="epilogue0",
        body=body,
        init=None
    )

    # Collect buffer_var_map
    buffer_var_map = {}

    def _collect_buffer_var_map(e):
        if isinstance(e, (tir.BufferLoad, tir.BufferStore)):
            buffer_var_map[e.buffer.data] = e.buffer

    tir.stmt_functor.post_order_visit(tmp_block.body, _collect_buffer_var_map)

    reads_writes = tir.analysis.get_block_read_write_region(tmp_block, buffer_var_map)

    epilogue_block = tir.Block(
        iter_vars=iter_vars,
        reads=reads_writes[0],
        writes=reads_writes[1],
        name_hint="epilogue0",
        body=body,
        init=None
    )

    return epilogue_block


def _build_loop_nest_for_block(
    block: tir.Block,
    loop_annotations: Dict[str, Dict[str, Any]],
) -> tir.Stmt:
    """Build For loop nest + BlockRealize for a block, ensuring each block's iter_vars are independent
    
    Loop order (from outer to inner):
    1. common spatial (with bind annotation, sorted by bind index)
    2. fused (with tag="fused" annotation)
    3. extra spatial (spatial dimensions without bind or fused tag)
    """
    from tvm.tir import stmt_functor
    from tvm import ir
    
    iter_vars = block.iter_vars
    
    # Step 1: Create fresh var for each iter_var to ensure independence
    var_remap = {}  # old_var -> fresh_var
    new_iter_vars = []
    for iv in iter_vars:
        fresh_var = tir.Var(iv.var.name, iv.var.dtype)
        var_remap[iv.var] = fresh_var
        new_iv = tir.IterVar(
            ir.Range(begin=0, end=iv.dom.extent),
            fresh_var,
            iv.iter_type,
            iv.thread_tag
        )
        new_iter_vars.append(new_iv)
    
    # Step 2: Use substitute to replace old variables in block body and init
    new_body = stmt_functor.substitute(block.body, var_remap)
    new_init = stmt_functor.substitute(block.init, var_remap) if block.init else None
    
    # Step 3: Replace old variables in reads/writes
    def _remap_buffer_regions(regions):
        new_regions = []
        for region in regions:
            new_ranges = []
            for r in region.region:
                new_min = stmt_functor.substitute(r.min, var_remap)
                new_extent = stmt_functor.substitute(r.extent, var_remap)
                new_ranges.append(ir.Range(new_min, new_min + new_extent))
            new_regions.append(tir.BufferRegion(region.buffer, new_ranges))
        return new_regions
    
    new_reads = _remap_buffer_regions(block.reads)
    new_writes = _remap_buffer_regions(block.writes)
    
    # Step 4: Create new block (using fresh vars)
    new_block = tir.Block(
        iter_vars=new_iter_vars,
        reads=new_reads,
        writes=new_writes,
        name_hint=block.name_hint,
        body=new_body,
        init=new_init,
        alloc_buffers=list(block.alloc_buffers) if block.alloc_buffers else [],
    )
    
    # Step 5: Reorder iter_vars to ensure correct loop nesting order
    # Expected order (outer to inner): common spatial (with bind) -> fused (with tag) -> extra spatial (no bind, no tag)
    # Note: loop_annotations key is original variable name (e.g., v_batch), needs matching
    
    common_spatial = []  # (idx, bind_index) - with bind annotation
    fused_dims = []      # idx - with tag="fused"
    extra_spatial = []   # idx - other spatial dimensions
    
    for i, iv in enumerate(new_iter_vars):
        var_name = iv.var.name
        ann = loop_annotations.get(var_name, {})
        
        if "bind" in ann:
            # Has bind annotation, is common spatial or splitp
            # Extract N from "vblockIdx.N" as sort key (larger N means more outer)
            bind_str = ann["bind"]
            if bind_str.startswith("vblockIdx."):
                bind_idx = int(bind_str.split(".")[-1])
            else:
                bind_idx = 0
            common_spatial.append((i, bind_idx))
        elif ann.get("tag") == "fused":
            # Has fused tag
            fused_dims.append(i)
        else:
            # Other (extra spatial)
            extra_spatial.append(i)
    
    # Sort common_spatial by bind_index descending (larger index means more outer)
    common_spatial.sort(key=lambda x: x[1], reverse=True)
    
    # Reorder: common_spatial -> fused -> extra_spatial
    sorted_indices = [idx for idx, _ in common_spatial] + fused_dims + extra_spatial
    
    sorted_iter_vars = [new_iter_vars[i] for i in sorted_indices]
    
    # Step 6: Create outer vars (remove v_ prefix) and arrange by new order
    sorted_outer_vars = []
    for iv in sorted_iter_vars:
        var_name = iv.var.name
        outer_name = var_name[2:] if var_name.startswith("v_") else var_name
        outer_var = tir.Var(outer_name, iv.var.dtype)
        sorted_outer_vars.append(outer_var)
    
    # Step 7: Create iter_values mapping (from sorted_outer_vars to new_iter_vars original order)
    # BlockRealize's iter_values must match block.iter_vars order
    # Need to establish mapping from original order to new order
    iter_values = []
    for iv in new_iter_vars:
        # Find corresponding position in sorted_iter_vars
        var_name = iv.var.name
        for j, sorted_iv in enumerate(sorted_iter_vars):
            if sorted_iv.var.name == var_name:
                iter_values.append(sorted_outer_vars[j])
                break
    
    block_realize = tir.BlockRealize(
        iter_values=iter_values,
        predicate=tir.const(True, "bool"),
        block=new_block
    )
    
    # Step 8: Build For loop nest in sorted order
    body = block_realize
    for iv, outer_var in zip(reversed(sorted_iter_vars), reversed(sorted_outer_vars)):
        extent = iv.dom.extent
        annotations = loop_annotations.get(iv.var.name, {})
        
        # Keep annotations as native Python types, no need to convert to TIR types
        # Schedule.annotate expects Python native types
        tir_annotations = {}
        for key, value in annotations.items():
            tir_annotations[key] = value
        
        body = tir.For(
            outer_var,
            tir.const(0, "int32"),
            extent if isinstance(extent, tir.PrimExpr) else tir.const(extent, "int32"),
            tir.ForKind.SERIAL,
            body,
            None,
            tir_annotations
        )
    
    return body


def transform_reductions_with_split(
    reduction_infos: CascadedGroupInfo,
    reduce_funcs_list: List[Tuple[BMat | None, BMat | None, BMat | None]],
    num_split: int
) -> tir.PrimFunc:
    """
    Transform cascaded reduction blocks into split version online algorithm form,
    programmatically construct Kernel1 PrimFunc.

    Args:
        reduction_infos: Cascaded reduction group info
        reduce_funcs_list: List of reduction functions
        num_split: Number of splits

    Returns:
        Kernel1 PrimFunc
    """
    assert num_split is not None and num_split > 0
    assert len(reduction_infos.reduction_configs) == len(reduce_funcs_list)

    n = len(reduction_infos.reduction_configs)
    sch = reduction_infos.sch

    # ===================== Phase A: Analysis and preparation =====================
    # Get function parameter buffers, distinguish input and original output
    func = sch.mod[sch.func_working_on]
    func_param_buffers = set()
    input_params = []  # Original input parameters
    input_buffers = []  # Original input buffers
    
    # First collect all buffers that are written to
    written_buffers = set()
    def collect_writes(stmt):
        if isinstance(stmt, tir.BlockRealize):
            block = stmt.block
            for buffer_region in block.writes:
                written_buffers.add(buffer_region.buffer.data)
    tir.stmt_functor.post_order_visit(func.body, collect_writes)

    # Only use read-only (not written) parameter buffers as kernel1 input
    for param in func.params:
        if param in func.buffer_map:
            buf = func.buffer_map[param]
            func_param_buffers.add(buf)
            # Only non-output buffers are kernel1 inputs
            if buf.data not in written_buffers:
                input_params.append(param)
                input_buffers.append(buf)

    # Get reduction axis info
    first_block_info = reduction_infos.cascaded_group[0]
    reduce_var_name = None
    reduce_extent = None
    for iter_info in first_block_info.iters:
        if iter_info.kind == "R":
            reduce_var_name = iter_info.var.name
            reduce_extent = iter_info.dom
            break
    assert reduce_var_name is not None and reduce_extent is not None

    chunk_size = reduce_extent // num_split

    # Create splitp variable
    splitp_var = tir.Var("v_splitp", "int32")

    # Get common spatial dimension info
    common_spatial_vars = []
    common_spatial_extents = []
    for iter_info in first_block_info.iters:
        if iter_info.kind == "S":
            common_spatial_vars.append(iter_info.var)
            common_spatial_extents.append(iter_info.dom)

    # ===================== Phase B: Construct buffer_remap =====================
    buffer_remap: Dict[tir.Buffer, tir.Buffer] = {}

    # Collect all internal buffers
    all_internal_buffers = set()
    for var_name, buffer_load in reduction_infos.x_buffer_load_map.items():
        if buffer_load.buffer not in func_param_buffers:
            all_internal_buffers.add(buffer_load.buffer)
    for var_name, buffer_load in reduction_infos.y_buffer_load_map.items():
        if buffer_load.buffer not in func_param_buffers:
            all_internal_buffers.add(buffer_load.buffer)

    # Identify x-type and y-type buffers
    x_type_buffers = set()
    for var_name, buffer_load in reduction_infos.x_buffer_load_map.items():
        if buffer_load.buffer not in func_param_buffers:
            x_type_buffers.add(buffer_load.buffer)

    y_type_buffers = set()
    for var_name, buffer_load in reduction_infos.y_buffer_load_map.items():
        if buffer_load.buffer not in func_param_buffers:
            y_type_buffers.add(buffer_load.buffer)

    # Create split version for each internal buffer
    for buf in all_internal_buffers:
        if buf in x_type_buffers:
            # x-type: Insert num_split before reduce axis
            # Need to find reduce axis position in buffer shape
            # Assume reduce axis is the last dimension (typical for flash attention)
            reduce_axis_pos = len(buf.shape) - 1
            new_shape = _create_split_buffer_shape(
                list(buf.shape), num_split, is_y_type=False, reduce_axis_pos=reduce_axis_pos
            )
        else:
            # y-type: Append num_split at end
            new_shape = _create_split_buffer_shape(
                list(buf.shape), num_split, is_y_type=True
            )

        split_buffer = tir.decl_buffer(
            shape=new_shape,
            dtype=buf.dtype,
            name=buf.name
        )
        buffer_remap[buf] = split_buffer

    # ===================== Phase C: Generate blocks =====================
    pro_ep_map = {}
    new_blocks = []
    extra_alloc_buffers = []

    # Generate reduction blocks
    for idx in range(n):
        new_block, extra_allocates = transform_single_block_with_split(
            idx,
            reduction_infos,
            reduce_funcs_list,
            pro_ep_map,
            num_split,
            func_param_buffers,
            reduce_var_name,
            reduce_extent,
            buffer_remap,
            chunk_size,
            splitp_var,
        )
        new_blocks.append(new_block)
        extra_alloc_buffers.extend(extra_allocates)

    # Generate prologue blocks
    prologue_blocks = []
    for ori_name, new_name in pro_ep_map.items():
        if not new_name.startswith("prologue"):
            continue
        # Get original prologue block
        ori_block = sch.get(sch.get_block(ori_name))

        prologue_block = transform_prologue_with_split(
            ori_block,
            num_split,
            func_param_buffers,
            reduce_var_name,
            buffer_remap,
            chunk_size,
            splitp_var,
            new_name,
        )
        prologue_blocks.append((new_name, prologue_block))

    # Create part_xxx output buffers
    part_output_buffers: Dict[str, tir.Buffer] = {}
    part_output_params = []

    for reduction_config in reduction_infos.reduction_configs:
        reduce_target_name = reduction_config.reduce_target
        reduce_target_buffer_load = reduction_infos.y_buffer_load_map[reduce_target_name]
        original_buffer = reduce_target_buffer_load.buffer
        split_buffer = buffer_remap.get(original_buffer, original_buffer)

        # part buffer shape same as split buffer, dtype is typically float16
        part_buffer = tir.decl_buffer(
            shape=split_buffer.shape,
            dtype="float16",  # Output is typically float16
            name=f"part_{original_buffer.name}"
        )
        part_output_buffers[reduce_target_name] = part_buffer

        # Create parameter Var
        part_param = tir.Var(f"part_{original_buffer.name}", "handle")
        part_output_params.append((part_param, part_buffer))

    # Collect extra spatial dimensions (get maximum from all reduction blocks)
    extra_spatial_vars = []
    extra_spatial_extents = []

    # Get extra spatial dimensions from second reduction block (e.g., head_dim_v)
    if n > 1:
        second_block_info = reduction_infos.cascaded_group[1]
        common_var_names = {v.name for v in common_spatial_vars}
        for iter_info in second_block_info.iters:
            if iter_info.kind == "S" and iter_info.var.name not in common_var_names:
                extra_spatial_vars.append(iter_info.var)
                extra_spatial_extents.append(iter_info.dom)

    # Generate epilogue block
    epilogue_block = build_epilogue_block(
        reduction_infos,
        buffer_remap,
        part_output_buffers,
        num_split,
        splitp_var,
        common_spatial_vars,
        common_spatial_extents,
        extra_spatial_vars,
        extra_spatial_extents,
    )

    # ===================== Phase D: Programmatically construct Kernel1 PrimFunc =====================

    # Determine loop annotations
    def _get_loop_annotations(block: tir.Block, is_prologue: bool = False) -> Dict[str, Dict[str, Any]]:
        """Get loop annotations for block"""
        annotations = {}
        bind_count = len(common_spatial_vars)  # Number of common spatial dimensions

        for i, iv in enumerate(block.iter_vars):
            var_name = iv.var.name
            # Remove v_ prefix, annotation name should match tile_map key
            base_name = var_name[2:] if var_name.startswith("v_") else var_name
            ann = {"name": base_name}

            # Check if this is a common spatial dimension
            is_common_spatial = any(v.name == var_name for v in common_spatial_vars)

            if is_common_spatial:
                # Calculate bind index
                idx = next(j for j, v in enumerate(common_spatial_vars) if v.name == var_name)
                ann["bind"] = f"vblockIdx.{bind_count - idx}"

            # splitp bind
            if var_name == "v_splitp":
                ann["bind"] = "vblockIdx.0"

            # Original cascade reduce axis tag (kv_len)
            # Note: Cannot use iter_type==2 to determine, because in prologue kv_len is Spatial while head_dim_qk is CommReduce
            # reduce_var_name itself may already have v_ prefix (e.g., "v_kv_len"), so compare directly
            if iv.var.name == reduce_var_name:
                ann["tag"] = "fused"

            annotations[var_name] = ann

        return annotations

    # Build loop nest for each block
    all_loop_nests = []

    # Prologue blocks
    for name, block in sorted(prologue_blocks, key=lambda x: x[0]):
        annotations = _get_loop_annotations(block, is_prologue=True)
        loop_nest = _build_loop_nest_for_block(block, annotations)
        all_loop_nests.append(loop_nest)

    # Reduction blocks
    for idx, block in enumerate(new_blocks):
        annotations = _get_loop_annotations(block)
        loop_nest = _build_loop_nest_for_block(block, annotations)
        all_loop_nests.append(loop_nest)

    # Epilogue block
    epilogue_annotations = _get_loop_annotations(epilogue_block)
    epilogue_loop_nest = _build_loop_nest_for_block(epilogue_block, epilogue_annotations)
    all_loop_nests.append(epilogue_loop_nest)

    # Assemble root block body
    root_body = tir.SeqStmt(all_loop_nests)

    # Collect all alloc buffers (deduplicated)
    all_alloc_buffers = []
    seen_buffer_names = set()
    # Split version internal buffers
    for buf in buffer_remap.values():
        if buf.name not in seen_buffer_names:
            all_alloc_buffers.append(buf)
            seen_buffer_names.add(buf.name)
    # Extra allocated buffers (prev_reduce, rescale_factor, input, etc.)
    for buf in extra_alloc_buffers:
        if buf.name not in seen_buffer_names:
            all_alloc_buffers.append(buf)
            seen_buffer_names.add(buf.name)

    # Create root block
    root_block = tir.Block(
        iter_vars=[],
        reads=[],
        writes=[],
        name_hint="root",
        body=root_body,
        init=None,
        alloc_buffers=all_alloc_buffers
    )

    # Create root BlockRealize
    func_body = tir.BlockRealize(
        iter_values=[],
        predicate=tir.const(True, "bool"),
        block=root_block
    )

    # Collect all buffers actually accessed in kernel1 body
    accessed_buffers = set()
    def _collect_accessed_buffers(stmt):
        if isinstance(stmt, (tir.BufferLoad, tir.BufferStore)):
            accessed_buffers.add(stmt.buffer)
    tir.stmt_functor.post_order_visit(root_body, _collect_accessed_buffers)

    # Assemble function parameters and buffer_map
    params = []
    buffer_map = {}

    # Input parameters: Only keep actually accessed buffers
    for param, buf in zip(input_params, input_buffers):
        if buf in accessed_buffers:
            params.append(param)
            buffer_map[param] = buf

    # Output parameters (part_xxx buffers)
    for param, buf in part_output_params:
        params.append(param)
        buffer_map[param] = buf

    # Create PrimFunc
    kernel1_func = tir.PrimFunc(
        params=params,
        body=func_body,
        buffer_map=buffer_map,
    )

    return kernel1_func, part_output_buffers


def build_kernel2_func(
    reduction_infos: CascadedGroupInfo,
    reduce_funcs_list: List[Tuple[BMat | None, BMat | None, BMat | None]],
    num_split: int,
    kernel1_part_buffers: Dict[str, tir.Buffer],
    kernel1_output_dtype: str = "float16",
) -> tir.PrimFunc:
    """
    Programmatically construct Kernel2 TIR PrimFunc.

    Kernel2 merges partial results from Kernel1, including:
    - reduction0: max reduction, get global maximum
    - reduction1..N-1: sum reduction, with rescale factor
    - epilogue: normalized output

    Args:
        reduction_infos: Cascaded reduction group info
        reduce_funcs_list: List of reduction functions
        num_split: Number of splits
        kernel1_part_buffers: Kernel1 output part buffer dict (reduce_target_name -> Buffer)
        kernel1_output_dtype: Final output dtype

    Returns:
        Kernel2 PrimFunc
    """
    n = len(reduction_infos.reduction_configs)
    assert n >= 2, "Kernel2 requires at least 2 reduction configs"

    # Helper function: Correctly handle variable names, avoid duplicate v_ prefix
    # This is consistent with Kernel1's _build_loop_nest_for_block naming strategy
    def _get_iter_var_name(var_name: str) -> str:
        """Convert variable name to v_xxx format, avoid duplicate v_ prefix"""
        base_name = var_name[2:] if var_name.startswith("v_") else var_name
        return f"v_{base_name}"

    def _get_base_name(var_name: str) -> str:
        """Extract base name of variable (remove v_ prefix)"""
        return var_name[2:] if var_name.startswith("v_") else var_name

    # ===================== Step A: Analysis and preparation =====================
    # Get first block info
    first_block_info = reduction_infos.cascaded_group[0]

    # Get common spatial dimensions (from first block)
    common_spatial_vars = []
    common_spatial_extents = []
    for iter_info in first_block_info.iters:
        if iter_info.kind == "S":
            common_spatial_vars.append(iter_info.var)
            common_spatial_extents.append(iter_info.dom)

    # Get extra spatial dimensions (find from all blocks the one with extra spatial dimensions)
    extra_spatial_vars = []
    extra_spatial_extents = []
    common_var_names = {v.name for v in common_spatial_vars}
    
    for block_info in reduction_infos.cascaded_group:
        for iter_info in block_info.iters:
            if iter_info.kind == "S" and iter_info.var.name not in common_var_names:
                # Check if this variable has already been added
                if iter_info.var.name not in {v.name for v in extra_spatial_vars}:
                    extra_spatial_vars.append(iter_info.var)
                    extra_spatial_extents.append(iter_info.dom)

    # Get first reduction's reduce_target_name (for T_all_max)
    first_reduce_target = reduction_infos.reduction_configs[0].reduce_target
    # Get second reduction's reduce_target_name (for T_all_exp_sum)
    second_reduce_target = reduction_infos.reduction_configs[1].reduce_target if n > 1 else None
    # Get last reduction's reduce_target_name (for T_all_o)
    last_reduce_target = reduction_infos.reduction_configs[n - 1].reduce_target

    # ===================== Step B: Create Kernel2 buffers =====================

    # Input buffers (from kernel1_part_buffers)
    part_buffers_ordered = []
    for config in reduction_infos.reduction_configs:
        reduce_target_name = config.reduce_target
        part_buf = kernel1_part_buffers[reduce_target_name]
        part_buffers_ordered.append((reduce_target_name, part_buf))

    # Output buffer shape: common_spatial_dimensions + extra_spatial_dimensions
    output_shape = list(common_spatial_extents) + list(extra_spatial_extents)
    output_buffer = tir.decl_buffer(
        shape=output_shape,
        dtype=kernel1_output_dtype,
        name="output"
    )

    # Internal alloc buffers
    all_alloc_buffers = []

    # T_all_xxx buffers (reduction results)
    t_all_buffers = {}
    for idx, config in enumerate(reduction_infos.reduction_configs):
        reduce_target_name = config.reduce_target
        part_buf = kernel1_part_buffers[reduce_target_name]
        
        # Determine if there are extra spatial dimensions by comparing part buffer shape
        # part buffer shape: [common_spatial..., extra_spatial..., num_split]
        # Last dimension is num_split
        part_shape_len = len(part_buf.shape)
        expected_len_no_extra = len(common_spatial_extents) + 1  # +1 for num_split
        has_extra_spatial = part_shape_len > expected_len_no_extra
        
        if has_extra_spatial:
            # Has extra spatial dimensions: Extract from part buffer shape
            # part buffer shape: [common_spatial..., extra_spatial..., num_split]
            # extra_spatial is after common_spatial, before num_split
            extra_extents = [part_buf.shape[i] for i in range(len(common_spatial_extents), part_shape_len - 1)]
            t_all_shape = list(common_spatial_extents) + extra_extents
        else:
            # No extra spatial dimensions: Only common spatial dimensions
            t_all_shape = list(common_spatial_extents)

        t_all_buf = tir.decl_buffer(
            shape=t_all_shape,
            dtype="float32",
            name=f"T_all_{reduce_target_name}"
        )
        t_all_buffers[reduce_target_name] = t_all_buf
        all_alloc_buffers.append(t_all_buf)

    # input_i buffers (Cast input values)
    input_buffers = {}
    for idx, (reduce_target_name, part_buf) in enumerate(part_buffers_ordered):
        input_buf = tir.decl_buffer(
            shape=list(part_buf.shape),
            dtype="float32",
            name=f"input_{idx}"
        )
        input_buffers[reduce_target_name] = input_buf
        all_alloc_buffers.append(input_buf)

    # rescale_factor_i buffers (only for reduction1 and later, not for reduction0)
    rescale_buffers = {}
    for idx, config in enumerate(reduction_infos.reduction_configs):
        if idx == 0:
            continue  # reduction0 does not need rescale factor
        reduce_target_name = config.reduce_target
        # rescale_factor shape: part_max shape (without extra spatial dimensions)
        part_max_shape = list(kernel1_part_buffers[first_reduce_target].shape)
        rescale_buf = tir.decl_buffer(
            shape=part_max_shape,
            dtype="float32",
            name=f"rescale_factor_{idx}"
        )
        rescale_buffers[reduce_target_name] = rescale_buf
        all_alloc_buffers.append(rescale_buf)

    # ===================== Step C: Construct reduction blocks =====================
    reduction_blocks = []

    # Get part_max buffer (for rescale calculation)
    part_max_buffer = kernel1_part_buffers[first_reduce_target]
    t_all_max_buffer = t_all_buffers[first_reduce_target]

    for idx, config in enumerate(reduction_infos.reduction_configs):
        reduce_target_name = config.reduce_target
        reduce_op = config.reduce_op
        part_buffer = kernel1_part_buffers[reduce_target_name]
        t_all_buffer = t_all_buffers[reduce_target_name]
        input_buffer = input_buffers[reduce_target_name]

        is_first = (idx == 0)
        is_last = (idx == n - 1)

        # Determine if current reduction has extra spatial dimensions
        # By comparing part buffer shape with common_spatial_dimensions + num_split
        part_shape_len = len(part_buffer.shape)
        expected_len_no_extra = len(common_spatial_extents) + 1  # +1 for num_split
        has_extra_spatial = part_shape_len > expected_len_no_extra

        # If there are extra spatial dimensions, infer from part buffer shape
        current_extra_spatial_vars = []
        current_extra_spatial_extents = []
        if has_extra_spatial:
            # part buffer shape: [common_spatial..., extra_spatial..., num_split]
            # extra_spatial is after common_spatial, before num_split
            num_extra = part_shape_len - expected_len_no_extra
            for i in range(num_extra):
                extra_extent = part_buffer.shape[len(common_spatial_extents) + i]
                # Use original extra spatial variable names (if available)
                if i < len(extra_spatial_vars):
                    current_extra_spatial_vars.append(extra_spatial_vars[i])
                    current_extra_spatial_extents.append(extra_extent)
                else:
                    # Create new variable
                    new_var = tir.Var(f"v_extra_{i}", "int32")
                    current_extra_spatial_vars.append(new_var)
                    current_extra_spatial_extents.append(extra_extent)

        # Create iter_vars
        iter_vars = []
        spatial_loop_vars = []
        reduce_loop_var = None

        # Common spatial dimensions
        for i, (var, extent) in enumerate(zip(common_spatial_vars, common_spatial_extents)):
            new_var = tir.Var(_get_iter_var_name(var.name), "int32")
            iv = tir.IterVar(
                dom=tvm.ir.Range(0, extent),
                var=new_var,
                iter_type=0,  # DataPar (Spatial)
            )
            iter_vars.append(iv)
            spatial_loop_vars.append((_get_base_name(var.name), new_var, extent))

        # Extra spatial dimensions (if current reduction has them)
        extra_iter_vars = []
        if has_extra_spatial:
            for var, extent in zip(current_extra_spatial_vars, current_extra_spatial_extents):
                var_name = var.name if hasattr(var, 'name') else str(var)
                new_var = tir.Var(_get_iter_var_name(var_name), "int32")
                iv = tir.IterVar(
                    dom=tvm.ir.Range(0, extent),
                    var=new_var,
                    iter_type=0,  # DataPar (Spatial)
                )
                iter_vars.append(iv)
                extra_iter_vars.append(iv)
                spatial_loop_vars.append((_get_base_name(var_name), new_var, extent))

        # splits reduction axis
        v_splits = tir.Var("v_splits", "int32")
        splits_iv = tir.IterVar(
            dom=tvm.ir.Range(0, num_split),
            var=v_splits,
            iter_type=2,  # CommReduce
        )
        iter_vars.append(splits_iv)
        reduce_loop_var = ("splits", v_splits, num_split)

        # Construct block body
        stmts = []

        # Common spatial indices
        common_spatial_indices = [iv.var for iv in iter_vars if iv.iter_type == 0][:len(common_spatial_vars)]
        # Extra spatial indices (if current reduction has them)
        current_extra_indices = [iv.var for iv in extra_iter_vars] if has_extra_spatial else []

        # part buffer indices: [common_spatial..., extra_spatial..., splits]
        if has_extra_spatial:
            part_indices = list(common_spatial_indices) + list(current_extra_indices) + [v_splits]
        else:
            part_indices = list(common_spatial_indices) + [v_splits]

        # T_all buffer indices: [common_spatial..., (extra_spatial...)]
        if has_extra_spatial:
            t_all_indices = list(common_spatial_indices) + list(current_extra_indices)
        else:
            t_all_indices = list(common_spatial_indices)

        # input buffer indices (same as part buffer)
        input_indices = part_indices.copy()

        if is_first:
            # reduction0: simple max reduction
            # input_0 = Cast("float32", part_max[...])
            cast_expr = tir.Cast("float32", tir.BufferLoad(part_buffer, part_indices))
            stmts.append(tir.BufferStore(input_buffer, cast_expr, input_indices))
            # T_all_max = max(T_all_max, input_0)
            t_all_load = tir.BufferLoad(t_all_buffer, t_all_indices)
            input_load = tir.BufferLoad(input_buffer, input_indices)
            max_expr = tir.Max(t_all_load, input_load)
            stmts.append(tir.BufferStore(t_all_buffer, max_expr, t_all_indices))

            # init statement: T_all_max = -inf
            init_value = tir.FloatImm("float32", float("-inf"))
            init_stmt = tir.BufferStore(t_all_buffer, init_value, t_all_indices)
        else:
            # reduction1+: sum reduction with rescale factor
            rescale_buffer = rescale_buffers[reduce_target_name]
            # rescale_factor indices (same as part_max, without extra spatial dimensions)
            rescale_indices = list(common_spatial_indices) + [v_splits]

            # rescale_factor = exp(Cast("float32", part_max[...]) - T_all_max[...])
            part_max_load = tir.BufferLoad(part_max_buffer, rescale_indices)
            t_all_max_load = tir.BufferLoad(t_all_max_buffer, common_spatial_indices)
            exp_arg = tir.Cast("float32", part_max_load) - t_all_max_load
            exp_expr = tir.call_intrin("float32", "tir.exp", exp_arg)
            stmts.append(tir.BufferStore(rescale_buffer, exp_expr, rescale_indices))

            # input = Cast("float32", part_xxx[...]) * rescale_factor
            part_load = tir.BufferLoad(part_buffer, part_indices)
            rescale_load = tir.BufferLoad(rescale_buffer, rescale_indices)
            input_expr = tir.Cast("float32", part_load) * rescale_load
            stmts.append(tir.BufferStore(input_buffer, input_expr, input_indices))

            # T_all_xxx = T_all_xxx + input
            t_all_load = tir.BufferLoad(t_all_buffer, t_all_indices)
            input_load = tir.BufferLoad(input_buffer, input_indices)
            sum_expr = t_all_load + input_load
            stmts.append(tir.BufferStore(t_all_buffer, sum_expr, t_all_indices))

            # init statement: T_all_xxx = 0.0
            init_value = tir.FloatImm("float32", 0.0)
            init_stmt = tir.BufferStore(t_all_buffer, init_value, t_all_indices)

        body = tir.SeqStmt(stmts) if len(stmts) > 1 else stmts[0]

        # Create temporary block to get reads/writes
        tmp_block = tir.Block(
            iter_vars=iter_vars,
            reads=[],
            writes=[],
            name_hint=f"reduction{idx}",
            body=body,
            init=init_stmt
        )

        # Collect buffer_var_map
        buffer_var_map = {}

        def _collect_buffer_var_map(e):
            if isinstance(e, (tir.BufferLoad, tir.BufferStore)):
                buffer_var_map[e.buffer.data] = e.buffer

        tir.stmt_functor.post_order_visit(tmp_block.body, _collect_buffer_var_map)
        tir.stmt_functor.post_order_visit(tmp_block.init, _collect_buffer_var_map)

        reads_writes = tir.analysis.get_block_read_write_region(tmp_block, buffer_var_map)

        reduction_block = tir.Block(
            iter_vars=iter_vars,
            reads=reads_writes[0],
            writes=reads_writes[1],
            name_hint=f"reduction{idx}",
            body=body,
            init=init_stmt
        )

        reduction_blocks.append(reduction_block)

    # ===================== Step D: Construct epilogue block =====================
    # epilogue: output = Cast("float16", T_all_o / T_all_exp_sum)
    
    # Find the T_all buffer with extra spatial dimensions (T_all_o)
    # and the T_all buffer of the last sum reduction without extra spatial dimensions (T_all_exp_sum)
    t_all_o_target = None
    t_all_exp_sum_target = None
    
    for config in reduction_infos.reduction_configs:
        reduce_target_name = config.reduce_target
        part_buf = kernel1_part_buffers[reduce_target_name]
        
        part_shape_len = len(part_buf.shape)
        expected_len_no_extra = len(common_spatial_extents) + 1
        has_extra_spatial = part_shape_len > expected_len_no_extra
        
        if has_extra_spatial:
            # Has extra spatial dimensions -> T_all_o
            t_all_o_target = reduce_target_name
        elif config.reduce_op == "+":
            # No extra spatial dimensions and sum reduction -> T_all_exp_sum
            t_all_exp_sum_target = reduce_target_name
    
    # If no reduction with extra spatial dimensions found, use the last reduction
    if t_all_o_target is None:
        t_all_o_target = last_reduce_target
    # If no exp_sum found, use the second reduction (if exists)
    if t_all_exp_sum_target is None:
        t_all_exp_sum_target = second_reduce_target if second_reduce_target else first_reduce_target

    epilogue_iter_vars = []

    # Common spatial dimensions
    for var, extent in zip(common_spatial_vars, common_spatial_extents):
        new_var = tir.Var(_get_iter_var_name(var.name), "int32")
        iv = tir.IterVar(
            dom=tvm.ir.Range(0, extent),
            var=new_var,
            iter_type=0,  # DataPar (Spatial)
        )
        epilogue_iter_vars.append(iv)

    # Extra spatial dimensions
    for var, extent in zip(extra_spatial_vars, extra_spatial_extents):
        new_var = tir.Var(_get_iter_var_name(var.name), "int32")
        iv = tir.IterVar(
            dom=tvm.ir.Range(0, extent),
            var=new_var,
            iter_type=0,  # DataPar (Spatial)
        )
        epilogue_iter_vars.append(iv)

    # Construct indices
    epilogue_spatial_indices = [iv.var for iv in epilogue_iter_vars]
    common_epilogue_indices = epilogue_spatial_indices[:len(common_spatial_vars)]

    # References to T_all_o and T_all_exp_sum buffers
    t_all_o_buffer = t_all_buffers[t_all_o_target]
    t_all_exp_sum_buffer = t_all_buffers[t_all_exp_sum_target]

    # output = Cast("float16", T_all_o / T_all_exp_sum)
    t_all_o_load = tir.BufferLoad(t_all_o_buffer, epilogue_spatial_indices)
    t_all_exp_sum_load = tir.BufferLoad(t_all_exp_sum_buffer, common_epilogue_indices)
    div_expr = t_all_o_load / t_all_exp_sum_load
    cast_expr = tir.Cast(kernel1_output_dtype, div_expr)
    epilogue_body = tir.BufferStore(output_buffer, cast_expr, epilogue_spatial_indices)

    # Create temporary epilogue block
    tmp_epilogue_block = tir.Block(
        iter_vars=epilogue_iter_vars,
        reads=[],
        writes=[],
        name_hint="epilogue0",
        body=epilogue_body,
        init=None
    )

    # Collect buffer_var_map
    buffer_var_map = {}

    def _collect_buffer_var_map_epilogue(e):
        if isinstance(e, (tir.BufferLoad, tir.BufferStore)):
            buffer_var_map[e.buffer.data] = e.buffer

    tir.stmt_functor.post_order_visit(tmp_epilogue_block.body, _collect_buffer_var_map_epilogue)

    reads_writes = tir.analysis.get_block_read_write_region(tmp_epilogue_block, buffer_var_map)

    epilogue_block = tir.Block(
        iter_vars=epilogue_iter_vars,
        reads=reads_writes[0],
        writes=reads_writes[1],
        name_hint="epilogue0",
        body=epilogue_body,
        init=None
    )

    # ===================== Step E: Construct PrimFunc =====================

    # Loop annotations
    def _get_kernel2_loop_annotations(block: tir.Block, is_epilogue: bool = False) -> Dict[str, Dict[str, Any]]:
        """Get loop annotations for Kernel2 block"""
        annotations = {}
        bind_count = len(common_spatial_vars)

        for iv in block.iter_vars:
            var_name = iv.var.name
            ann = {"name": _get_base_name(var_name)}

            # Check if it's a common spatial dimension
            # Use _get_iter_var_name to correctly match variable names
            is_common_spatial = any(_get_iter_var_name(v.name) == var_name for v in common_spatial_vars)

            if is_common_spatial:
                # Calculate bind index
                idx = next(j for j, v in enumerate(common_spatial_vars) if _get_iter_var_name(v.name) == var_name)
                ann["bind"] = f"vblockIdx.{bind_count - 1 - idx}"

            annotations[var_name] = ann

        return annotations

    # Construct loop nest for each block
    all_loop_nests = []

    # Reduction blocks (each with independent loop nest)
    for block in reduction_blocks:
        annotations = _get_kernel2_loop_annotations(block)
        loop_nest = _build_loop_nest_for_block(block, annotations)
        all_loop_nests.append(loop_nest)

    # Epilogue block
    epilogue_annotations = _get_kernel2_loop_annotations(epilogue_block, is_epilogue=True)
    epilogue_loop_nest = _build_loop_nest_for_block(epilogue_block, epilogue_annotations)
    all_loop_nests.append(epilogue_loop_nest)

    # Assemble root block body
    root_body = tir.SeqStmt(all_loop_nests)

    # Create root block
    root_block = tir.Block(
        iter_vars=[],
        reads=[],
        writes=[],
        name_hint="root",
        body=root_body,
        init=None,
        alloc_buffers=all_alloc_buffers
    )

    # Create root BlockRealize
    func_body = tir.BlockRealize(
        iter_values=[],
        predicate=tir.const(True, "bool"),
        block=root_block
    )

    # Assemble function parameters and buffer_map
    params = []
    buffer_map = {}

    # Input parameters (part buffers, in reduction_configs order)
    for reduce_target_name, part_buf in part_buffers_ordered:
        part_param = tir.Var(part_buf.name, "handle")
        params.append(part_param)
        buffer_map[part_param] = part_buf

    # Output parameter
    output_param = tir.Var(output_buffer.name, "handle")
    params.append(output_param)
    buffer_map[output_param] = output_buffer

    # Create PrimFunc
    kernel2_func = tir.PrimFunc(
        params=params,
        body=func_body,
        buffer_map=buffer_map,
    )

    return kernel2_func
