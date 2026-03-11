import tvm
from tvm import te, topi

from tvm.redfuser import (
    DecomposeReduction,
    UnifyBindOuterLoops,
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
    PromoteExternReduction
)
from tvm.target.codegen import function_to_tilelang_script
from pathlib import Path
current_dir = Path(__file__).parent


def redfuser_moe_routing(A, topk, axis=-1):
    max_elem, _, exp_sum = topi.nn.softmax_split(A, axis, reduce_max_name="k", reduce_sum_name="k", varargs_names=["m"]) # max_elem, exp_sum: float32
    topk_values, topk_indices = topi.tir_reduce_topk(A, topk, axis, reduce_topk_name="k", varargs_names=["m"])

    topk_values = te.compute(
        topk_values.shape,
        lambda *indices: te.exp(topk_values(*indices).astype("float32") - max_elem(*indices[:1])) / exp_sum(*indices[:1]),
        name="T_values_fp32", # 这里的名字不太重要,会被inline消除掉
        varargs_names=["m", "topk"]
    )

    return te.compute(topk_values.shape, lambda *indices: topk_values(*indices).astype(A.dtype), name="T_values", varargs_names=["m", "topk"]), te.compute(topk_indices.shape, lambda *indices: topk_indices(*indices), name="T_indices", varargs_names=["m", "topk"])


def main(func_name, tile_map):
    A = te.placeholder([4096, 4096], "float16", name="A")
    topk_values, topk_indices = redfuser_moe_routing(A, 8)
    func = te.create_prim_func([A, topk_values, topk_indices])

    mod = tvm.IRModule({func_name: func})
    
    passes = tvm.transform.Sequential([
        # process intrin
        PromoteExternReduction,
        # generate online expr
        GenerateOnlineExpr(),
        # tiling
        UnifyBindOuterLoops,
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

    with open(current_dir.joinpath("../utils", f"tilelang_funcs.py"), "r") as f:
        import_stmt = f.read() + "\n\n"
    tilelang_prog = import_stmt  + function_to_tilelang_script(func_name, mod[func_name])
    print(tilelang_prog, file=open(current_dir.joinpath("generated", f"generated_{func_name}.py"), "w"))
    print(f"Generated code saved to {current_dir.joinpath('generated', f'generated_{func_name}.py')}")


if __name__ == "__main__":
    tile_map = {"m": 128, "k": 128}
    main("redfuser_moe_routing", tile_map)
