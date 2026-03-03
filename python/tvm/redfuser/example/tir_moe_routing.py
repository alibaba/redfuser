import tvm
from tvm import te, topi
from tvm.script import ir as I
from tvm.script import tir as T

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
)
from tvm.target.codegen import function_to_tilelang_script
from pathlib import Path
current_dir = Path(__file__).parent


def redfuser_moe_routing(X, topk, axis = -1):
    # topk_values, topk_indices = topi.topk(X, k = 8, axis = -1, dtype="int32")
    topk_values, topk_indices = topi.tir_reduce_topk(X, topk, axis)

    return topk_values, topk_indices

@I.ir_module
class Module:
    @T.prim_func
    def redfuser_moe_routing(
        A: T.Buffer((4096, 4096), "float16"), 
        T_values: T.Buffer((4096, 8), "float16"),
        T_indices: T.Buffer((4096, 8), "int32"),
    ):
        T.func_attr({"global_symbol": "main", "layout_free_buffers": [1], "tir.noalias": True})
        # with T.block("root"):
        input0 = T.alloc_buffer((4096, 4096))
        T_max_elem = T.alloc_buffer((4096,))
        prev_T_max_elem = T.alloc_buffer((4096,))

        input1 = T.alloc_buffer((4096, 4096))
        T_topk_elem = T.alloc_buffer((4096, 8))
        T_topk_indices = T.alloc_buffer((4096, 8), "int32")

        rescale_factor_2 = T.alloc_buffer((4096,))
        input2 = T.alloc_buffer((4096, 4096))
        T_exp_sum = T.alloc_buffer((4096,))
        for m in T.serial(4096, annotations={"bind": "vblockIdx.0", "name": "m"}):
            for k in T.serial(4096, annotations={"name": "k", "tag": "fused"}):
                with T.block("reduction0"):
                    v_m, v_k = T.axis.remap("SR", [m, k])
                    T.reads(A[v_m, v_k])
                    T.writes(input0[v_m, v_k], prev_T_max_elem[v_m], T_max_elem[v_m])
                    # T.writes(input0[v_m, v_k * 8 : v_k * 8 + 8], prev_T_max_elem[v_m], T_max_elem[v_m])
                    with T.init():
                        T_max_elem[v_m] = T.float32(-1e5)
                    prev_T_max_elem[v_m] = T_max_elem[v_m]
                    input0[v_m, v_k] = T.Cast("float32", A[v_m, v_k])
                    T_max_elem[v_m] = T.max(T_max_elem[v_m], input0[v_m, v_k])
        for m in T.serial(4096, annotations={"bind": "vblockIdx.0", "name": "m"}):
            for k in T.serial(4096, annotations={"name": "k", "tag": "fused"}):
                with T.block("reduction1"):
                    v_m, v_k = T.axis.remap("SR", [m, k])
                    T.reads(A[v_m, v_k])
                    T.writes(input1[v_m, v_k], T_topk_elem[v_m, 0:8], T_topk_indices[v_m, 0:8])
                    with T.init():
                        T_topk_elem[v_m, 0:8] = T.Broadcast(T.float32(-1e5), 8)
                        T_topk_indices[v_m, 0:8] = T.Broadcast(T.int32(-1), 8)
                    input1[v_m, v_k] = T.Cast("float32", A[v_m, v_k])
                    T.evaluate(T.call_intrin(
                        "handle",
                        "tir.vec_reduce",
                        "topk",
                        8,
                        -1,
                        input1[v_m, v_k],
                        T_topk_elem[v_m, 0:8],
                        T_topk_indices[v_m, 0:8],
                        v_k
                    ))

        for m in T.serial(4096, annotations={"bind": "vblockIdx.0", "name": "m"}):
            for k in T.serial(4096, annotations={"name": "k", "tag": "fused"}):
                with T.block("reduction2"):
                    v_m, v_k = T.axis.remap("SR", [m, k])
                    T.reads(prev_T_max_elem[v_m], T_max_elem[v_m], A[v_m, v_k])
                    T.writes(rescale_factor_2[v_m], input2[v_m, v_k], T_exp_sum[v_m])
                    with T.init():
                        T_exp_sum[v_m] = T.float32(0.0)
                    rescale_factor_2[v_m] = T.exp(T.float32(-1.0) * T_max_elem[v_m] + prev_T_max_elem[v_m])
                    input2[v_m, v_k] = T.exp(T.Cast("float32", A[v_m, v_k]) - T_max_elem[v_m])
                    T_exp_sum[v_m] =  T_exp_sum[v_m] * rescale_factor_2[v_m]
                    T_exp_sum[v_m] = T_exp_sum[v_m] + input2[v_m, v_k]
                    
        for m in T.serial(4096, annotations={"bind": "vblockIdx.0", "name": "m"}):
            for topk in T.serial(8, annotations={"name": "topk"}):
                with T.block("epilogue0"):
                    v_m, v_k = T.axis.remap("SS", [m, topk])
                    T.reads(T_topk_elem[v_m, v_k], T_topk_indices[v_m, v_k], T_exp_sum[v_m], T_max_elem[v_m])
                    T.writes(T_values[v_m, v_k], T_indices[v_m, v_k])
                    T_values[v_m, v_k] = T.Cast("float16", T.exp(T_topk_elem[v_m, v_k] - T_max_elem[v_m]) / T_exp_sum[v_m])
                    T_indices[v_m, v_k] = T_topk_indices[v_m, v_k]


def main(func_name, tile_map):
    # X = te.placeholder([4096, 4096], "float16", name="X")
    # topk_values, topk_indices = redfuser_moe_routing(X, 8)
    # func = te.create_prim_func([X, topk_values, topk_indices])

    # mod = tvm.IRModule({func_name: func})
    mod = Module
    # mod.show()
    
    passes = tvm.transform.Sequential([
        # generate online expr
        # GenerateOnlineExpr,
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
        ]
    )

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
