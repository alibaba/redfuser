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
    UnifyBindOuterLoops,
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
    _, exp, exp_sum = topi.nn.softmax_split(topi.multiply(s, scale), reduce_max_name="kv_len", reduce_sum_name="kv_len", varargs_names=["batch", "head_num", "q_len"])
    # o.shape = [bs, hn, ql, hd]
    o = topi.nn.matmul(exp.astype(dtype), v, out_dtype=accum_dtype, reduce_axis_name="kv_len", varargs_names=["batch", "head_num", "q_len", "head_dim_v"])
    o_norm = te.compute(o.shape, lambda *indices: o(*indices) / exp_sum(*indices[:-1]), name="T_softmax_norm", varargs_names=["batch", "head_num", "q_len", "head_dim_v"])

    return te.compute(o_norm.shape, lambda *indices: o_norm(*indices).astype(dtype), name="T_Cast", varargs_names=["batch", "head_num", "q_len", "head_dim_v"])


def main(tile_map, num_split=4):
    func_name = "redfuser_flash_decoding"
    
    q = te.placeholder([128, 16, 512, 64], "float16", name="q")
    k = te.placeholder([128, 16, 512, 64], "float16", name="k")
    v = te.placeholder([128, 16, 512, 64], "float16", name="v")
    o = redfuser_flash_attention(q, k, v, dtype="float16", accum_dtype="float32")
    func = te.create_prim_func([q, k, v, o])
    mod = tvm.IRModule({func_name: func})

    passes = tvm.transform.Sequential([
        # generate online expr with split
        GenerateOnlineExpr(num_split=num_split),
        # unify bind outer loops
        UnifyBindOuterLoops,
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
        BindBlockIdx
    ])

    mod = passes(mod)
    mod.show()

    # Generate tilelang code for two kernels
    import_stmt = "import tilelang\nimport tilelang.language as T\n\n"
    kernel1_name = f"{func_name}_kernel1"
    kernel2_name = f"{func_name}_kernel2"
    tilelang_prog = import_stmt + \
        function_to_tilelang_script("redfuser_flash_decoding1", mod[kernel1_name]) + \
        function_to_tilelang_script("redfuser_flash_decoding2", mod[kernel2_name])
    
    output_file = current_dir.joinpath("generated", "generated_redfuser_flash_decoding.py")
    print(tilelang_prog, file=open(output_file, "w"))
    print(f"Generated code saved to {output_file}")


if __name__ == "__main__":
    tile_map = {"q_len": 64, "kv_len": 64, "head_dim_v": 64, "head_dim_qk": 64, "splits": 1}
    main(tile_map)
