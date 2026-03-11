# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# pylint: disable=too-many-arguments
"""Argsort operator"""
import tvm
from tvm import te
from .utils import get_const_tuple


def sort(data, axis=-1, is_ascend=1):
    """Performs sorting along the given axis and returns an array
    in sorted order.

    Parameters
    ----------
    data : tvm.te.Tensor
        The input tensor.

    axis : int, optional
        Axis along which to sort the input tensor.
        By default the flattened array is used.

    is_ascend : boolean, optional
        Whether to sort in ascending or descending order.

    dtype : string, optional
        DType of the output indices.

    Returns
    -------
    out : tvm.te.Tensor
        Sorted index tensor.

    """
    data_buf = tvm.tir.decl_buffer(data.shape, data.dtype, "data_buf", data_alignment=8)
    out_buf = tvm.tir.decl_buffer(data.shape, data.dtype, "out_buf", data_alignment=8)
    out = te.extern(
        data.shape,
        [data],
        lambda ins, outs: tvm.tir.call_packed(
            "tvm.contrib.sort.sort", ins[0], outs[0], axis, is_ascend
        ),
        dtype=data.dtype,
        in_buffers=[data_buf],
        out_buffers=out_buf,
        name="sort_cpu",
        tag="sort_cpu",
    )
    return out


def argsort(data, valid_count=None, axis=-1, is_ascend=1, dtype="float32"):
    """Performs sorting along the given axis and returns an array
    of indices having the same shape as an input array that index
    data in sorted order.

    Parameters
    ----------
    data : tvm.te.Tensor
        The input tensor.

    valid_count : tvm.te.Tensor, optional
        1-D tensor for valid number of boxes.

    axis : int, optional
        Axis along which to sort the input tensor.
        By default the flattened array is used.

    is_ascend : boolean, optional
        Whether to sort in ascending or descending order.

    dtype : string, optional
        DType of the output indices.

    Returns
    -------
    out : tvm.te.Tensor
        Sorted index tensor.

    Example
    --------
    .. code-block:: python

        # An example to use argsort
        dshape = (1, 5, 6)
        data = te.placeholder(dshape, name="data")
        axis = 0
        is_ascend = False
        out = argsort(data, axis=axis, is_ascend=is_ascend)
        np_data = np.random.uniform(dshape)
        s = topi.generic.schedule_argsort(out)
        f = tvm.compile(s, [data, out], "llvm")
        dev = tvm.cpu()
        tvm_data = tvm.runtime.tensor(np_data, dev)
        tvm_out = tvm.runtime.tensor(np.zeros(dshape, dtype=data.dtype), dev)
        f(tvm_data, tvm_out)
    """
    data_buf = tvm.tir.decl_buffer(data.shape, data.dtype, "data_buf", data_alignment=8)
    if valid_count is not None:
        valid_count_buf = tvm.tir.decl_buffer(
            valid_count.shape, valid_count.dtype, "valid_count_buf", data_alignment=4
        )
        out_buf = tvm.tir.decl_buffer(data.shape, "int32", "out_buf", data_alignment=8)
        out = te.extern(
            data.shape,
            [data, valid_count],
            lambda ins, outs: tvm.tir.call_packed(
                "tvm.contrib.sort.argsort_nms", ins[0], ins[1], outs[0], axis, is_ascend
            ),
            dtype="int32",
            in_buffers=[data_buf, valid_count_buf],
            out_buffers=out_buf,
            name="argsort_nms_cpu",
            tag="argsort_nms_cpu",
        )
    else:
        out_buf = tvm.tir.decl_buffer(data.shape, dtype, "out_buf", data_alignment=8)
        out = te.extern(
            data.shape,
            [data],
            lambda ins, outs: tvm.tir.call_packed(
                "tvm.contrib.sort.argsort", ins[0], outs[0], axis, is_ascend
            ),
            dtype=dtype,
            in_buffers=[data_buf],
            out_buffers=out_buf,
            name="argsort_cpu",
            tag="argsort_cpu",
        )
    return out


def topk(data, k=1, axis=-1, ret_type="both", is_ascend=False, dtype="int64"):
    """Get the top k elements in an input tensor along the given axis.

    Parameters
    ----------
    data : tvm.te.Tensor
        The input tensor.

    k : int or tvm.te.Tensor, optional
        Number of top elements to select. Return all elements if k < 1.

    axis : int, optional
        Axis long which to sort the input tensor.

    ret_type: str, optional
        The return type [both, values, indices].
        "both": return both top k data and indices.
        "values": return top k data only.
        "indices": return top k indices only.

    is_ascend : boolean, optional
        Whether to sort in ascending or descending order.

    dtype : string, optional
        The data type of the indices output.

    Returns
    -------
    out : tvm.te.Tensor or List[tvm.te.Tensor]
        The computed result.
    """
    assert ret_type in ["both", "values", "indices"]
    data_buf = tvm.tir.decl_buffer(data.shape, data.dtype, "data_buf", data_alignment=8)
    out_shape = list(get_const_tuple(data.shape))
    kvar = tvm.te.size_var("k")
    if not isinstance(k, int):
        out_shape[axis] = kvar
    elif k >= 1:
        out_shape[axis] = k
    out_bufs = []
    if ret_type in ["both", "values"]:
        out_bufs.append(tvm.tir.decl_buffer(out_shape, data.dtype, "value_buf", data_alignment=8))
    if ret_type in ["both", "indices"]:
        out_bufs.append(tvm.tir.decl_buffer(out_shape, dtype, "indices_buf", data_alignment=8))
    out_shapes = [out_shape] * len(out_bufs)

    kv = kvar if not isinstance(k, int) else k
    out = te.extern(
        out_shapes,
        [data],
        lambda ins, outs: tvm.tir.call_packed(
            "tvm.contrib.sort.topk", ins[0], *outs, kv, axis, ret_type, is_ascend
        ),
        in_buffers=[data_buf],
        out_buffers=out_bufs,
        name="topk_cpu",
        tag="topk_cpu",
    )
    return out


def tir_reduce_topk(data, k=1, axis=-1, reduce_topk_name="k", varargs_names=None):
    """Get the top k elements in an input tensor along the given axis. This is just a declaration of the builtin function.

    Parameters
    ----------
    data : tvm.te.Tensor
        The input tensor.

    k : int or tvm.te.Tensor, optional
        Number of top elements to select. Return all elements if k < 1.

    axis : int, optional
        Axis long which to sort the input tensor.

    Returns
    -------
    out : tvm.te.Tensor or List[tvm.te.Tensor]
        The computed result.
    """

    assert isinstance(k, int) and k >= 1

    out_shape = list(get_const_tuple(data.shape))
    ndim = len(out_shape)
    real_axis = axis if axis >= 0 else axis + ndim
    out_shape[real_axis] = k

    if varargs_names is not None:
        assert len(varargs_names) == ndim - 1
        varargs_names.insert(real_axis, None) # real_axis这个索引不会被索引到(因为会变成"k"),但是这里要把varargs_names补齐,类似这种case: m, k, n -> SRS

    out_bufs = [
        tvm.tir.decl_buffer(out_shape, "float32", "T_topk_elem"), # FIXME(liyangcheng): dtype issue
        tvm.tir.decl_buffer(out_shape, "int32", "T_topk_indices")
    ]
    out_shapes = [out_shape, out_shape]

    # Build the full shape of the reduce axis
    reduce_extent = get_const_tuple(data.shape)[real_axis]

    def __call_intrin(ins, outs):
        ib = tvm.tir.ir_builder.create()
        data_buf = ins[0]       # Buffer with full shape, e.g. [M, K]
        values_buf = outs[0]    # Buffer with output shape, e.g. [M, k]
        indices_buf = outs[1]   # Buffer with output shape, e.g. [M, k]

        # Build loops over all axes except the reduce axis
        # For the common 2D case: outer dims are all dims except real_axis
        shape = data_buf.shape

        def _build_loops(dim, loop_vars):
            if dim == ndim:
                # All loops built; now build the reduce loop
                k_var = tvm.tir.Var(reduce_topk_name, "int32")
                # Construct indices for data_buf: replace real_axis with k_var
                data_indices = list(loop_vars)
                data_indices[real_axis] = k_var
                # Construct indices for values/indices buf: replace real_axis with Ramp(0,1,k)
                out_indices = list(loop_vars)
                out_indices[real_axis] = tvm.tir.Ramp(tvm.tir.const(0, "int32"), tvm.tir.const(1, "int32"), k)

                stmt = tvm.tir.For(
                    k_var, tvm.tir.const(0, "int32"), tvm.tir.const(reduce_extent, "int32"),
                    tvm.tir.ForKind.SERIAL,
                    tvm.tir.Evaluate(tvm.tir.call_intrin(
                        "handle",
                        tvm.tir.op.Op.get("tir.vec_reduce"),
                        "topk",
                        k,
                        axis,
                        tvm.tir.BufferLoad(data_buf, data_indices),
                        tvm.tir.BufferLoad(values_buf, out_indices),
                        tvm.tir.BufferLoad(indices_buf, out_indices),
                        k_var,
                    ))
                )
                return stmt
            else:
                if dim == real_axis:
                    # Skip real_axis here; it becomes the reduce loop variable
                    loop_vars.append(None)
                    inner = _build_loops(dim + 1, loop_vars)
                    loop_vars.pop()
                    return inner
                else:
                    outer_var = tvm.tir.Var(f"{varargs_names[dim] if varargs_names is not None else f'i{dim}'}", "int32") # len(varargs_name) == (ndim-1)+1,如果外面传对的话,这里应该不会出现索引的问题
                    loop_vars.append(outer_var)
                    inner = _build_loops(dim + 1, loop_vars)
                    loop_vars.pop()
                    return tvm.tir.For(
                        outer_var, tvm.tir.const(0, "int32"), shape[dim],
                        tvm.tir.ForKind.SERIAL,
                        inner
                    )

        body = _build_loops(0, [])
        ib.emit(body)
        return ib.get()

    out = te.extern(
        out_shapes,
        [data],
        __call_intrin,
        out_buffers=out_bufs,
        name="tir_reduce_topk",
        tag="tir_reduce_topk"
    )

    return out
