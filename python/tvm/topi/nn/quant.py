"""TVM operator for quant compute."""
from __future__ import absolute_import
import tvm
from tvm import te, topi

def per_axis_fp8_quant(x, axis = -1, reduce_axis_name="k", varargs_names=[]):
    FP8_MAX = 448.0
    shape = x.shape
    if axis < 0:
        axis = len(shape) + axis
    if axis >= len(shape):
        ValueError("axis parameter should be less than input dim")

    allargs_names = varargs_names[:axis] + [reduce_axis_name, ] + varargs_names[axis:]
    k = te.reduce_axis((0, shape[axis]), name=reduce_axis_name)

    def insert_reduce_index(indices, reduce_index):
        return indices[:axis] + (reduce_index,) + indices[axis:]

    def get_non_reduce_indices(indices):
        return tuple([var for (i, var) in enumerate(indices) if i != axis])

    def _compute_max(abs_elem, *indices):
        eval_range = insert_reduce_index(indices, k)
        return tvm.te.max(abs_elem[eval_range], axis=k).astype("float32")

    def _compute_quant(scale_elem, *indices):
        non_reduce_indices = get_non_reduce_indices(indices)
        return tvm.te.div(x[indices], scale_elem[non_reduce_indices]).astype("float8_e4m3fn")

    reduced_shape = get_non_reduce_indices(shape)
    abs_elem = te.compute(shape, lambda *indices: tvm.te.abs(x[indices]).astype("float32"), name="T_abs_elem", varargs_names=allargs_names)
    max_elem = te.compute(reduced_shape, lambda *indices: _compute_max(abs_elem, *indices), name="T_max_elem", varargs_names=varargs_names)
    scale_elem = te.compute(reduced_shape, lambda *indices: tvm.te.div(max_elem[indices], FP8_MAX), name="T_scale_elem", varargs_names=varargs_names)
    quant_elem = te.compute(shape, lambda *indices: _compute_quant(scale_elem, *indices), name="T_quant_elem", varargs_names=allargs_names)

    return quant_elem, scale_elem