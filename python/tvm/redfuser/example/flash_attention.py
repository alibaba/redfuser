import tvm
from tvm import te, topi
from tvm.redfuser import (
    DecomposeReduction,
    TileByAnnotation,
    BlockizeInnerLoops,
    MergeFusedLoops,
    EliminateUnitLoops,
    TransformIOBuffers,
    SetBufferScope,
    HoistTLCopy,
    BindBlockIdx,
    MoveAllocBuffer,
    UnifyGemmDtype,
    GenerateOnlineExpr,
)
from tvm.target.codegen import function_to_tilelang_script
from pathlib import Path
current_dir = Path(__file__).parent


def redfuser_flash_attention(q, k, v, dtype, accum_dtype):
    hd = q.shape[-1].astype("float32")
    scale = te.div(1.0, te.sqrt(hd))

    # s.shape = [bs, hn, ql, kvl]
    s = topi.nn.matmul(q, k, transpose_b=True, out_dtype=accum_dtype, reduce_axis_name="head_dim_qk", varargs_names=["batch", "head_num", "q_len", "kv_len"])
    # exp.shape = [bs, hn, ql, kvl], exp_sum.shape = [bs, hn, ql]
    exp, exp_sum = topi.nn.softmax_split(topi.multiply(s, scale), reduce_max_name="kv_len", reduce_sum_name="kv_len", varargs_names=["batch", "head_num", "q_len"])
    # o.shape = [bs, hn, ql, hd]
    o = topi.nn.matmul(exp.astype(dtype), v, out_dtype=accum_dtype, reduce_axis_name="kv_len", varargs_names=["batch", "head_num", "q_len", "head_dim_v"])
    o_norm = te.compute(o.shape, lambda *indices: o(*indices) / exp_sum(*indices[:-1]), name="T_softmax_norm", varargs_names=["batch", "head_num", "q_len", "head_dim_v"])

    return te.compute(o_norm.shape, lambda *indices: o_norm(*indices).astype(dtype), name="T_Cast", varargs_names=["batch", "head_num", "q_len", "head_dim_v"])


def main(func_name, tile_map):
    q = te.placeholder([128, 16, 512, 64], "float16", name="q")
    k = te.placeholder([128, 16, 512, 64], "float16", name="k")
    v = te.placeholder([128, 16, 512, 64], "float16", name="v")
    o = redfuser_flash_attention(q, k, v, dtype="float16", accum_dtype="float32")
    func = te.create_prim_func([q, k, v, o])
    mod = tvm.IRModule({func_name: func})

    passes = tvm.transform.Sequential([
        # generate online expr
        GenerateOnlineExpr,
        # tiling
        TileByAnnotation(tile_map),
        # eliminate unit loops and merge fused loops
        EliminateUnitLoops,
        MergeFusedLoops,
        # cache IO buffers and set buffer scope
        TransformIOBuffers,
        SetBufferScope,
        # blockize inner loops
        DecomposeReduction,
        BlockizeInnerLoops,
        # compact alloc_buffer size
        tvm.tir.transform.CompactBufferAllocation(
            is_strict=True, remove_trivial_dims=True
        ),
        # convert to tilelang builtins
        tvm.tir.transform.ConvertToTileLangBuiltins(),
        UnifyGemmDtype,
        HoistTLCopy,
        MoveAllocBuffer,
        BindBlockIdx]
    )

    mod = passes(mod)

    tilelang_prog = function_to_tilelang_script(func_name, mod[func_name])
    print(tilelang_prog, file=open(current_dir.joinpath("generated", f"generated_{func_name}.py"), "w"))
    print(f"Generated code saved to {current_dir.joinpath('generated', f'generated_{func_name}.py')}")


if __name__ == "__main__":
    tile_map = {"q_len": 64, "kv_len": 64, "head_dim_v": 64, "head_dim_qk": 64}
    main("redfuser_flash_attention", tile_map)
