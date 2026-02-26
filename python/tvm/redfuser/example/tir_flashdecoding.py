import tvm

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

@I.ir_module
class Module:
    @T.prim_func
    def redfuser_flash_decoding1(
        q: T.Buffer((128, 16, 512, 64), "float16"), 
        k: T.Buffer((128, 16, 512, 64), "float16"), 
        v: T.Buffer((128, 16, 512, 64), "float16"), 
        part_max: T.Buffer((128, 16, 512, 4), "float16"), # batch, head_num, q_len, num_split
        part_exp_sum: T.Buffer((128, 16, 512, 4), "float16"), # batch, head_num, q_len, num_split
        part_output: T.Buffer((128, 16, 512, 4, 64), "float16") # batch, head_num, q_len, num_split, head_dim_v
    ): 
        # shape: 
        # kv_len 被view为 num_split, ceil_div(kv_len, num_split)，包括中间变量，输出等Buffer
        # 所有reduce结果都增加一个num_split维度
        # 输入Buffer的kv_len维度需要修改为 num_split * ceil_div(kv_len, num_split) + kv_len
        # parallel dim:
        # 增加一个split的并行维度
        T_matmul_NT = T.alloc_buffer((128, 16, 512, 4, 128)) # batch, head_num, q_len, num_split, ceil_div(kv_len, num_split)
        T_softmax_maxelem = T.alloc_buffer((128, 16, 512, 4)) # batch, head_num, q_len, num_split
        T_matmul_NN = T.alloc_buffer((128, 16, 512, 4, 64))  # batch, head_num, q_len, num_split, head_dim_v
        T_softmax_expsum = T.alloc_buffer((128, 16, 512, 4)) # batch, head_num, q_len, num_split
        prev_T_softmax_maxelem = T.alloc_buffer((128, 16, 512, 4)) # batch, head_num, q_len, num_split
        input_0 = T.alloc_buffer((128, 16, 512, 4, 128)) # batch, head_num, q_len, num_split, ceil_div(kv_len, num_split)
        rescale_factor_1 = T.alloc_buffer((128, 16, 512, 4)) # batch, head_num, q_len, num_split
        input_1 = T.alloc_buffer((128, 16, 512, 4, 128)) # batch, head_num, q_len, num_split, ceil_div(kv_len, num_split)
        rescale_factor_2 = T.alloc_buffer((128, 16, 512, 4)) # batch, head_num, q_len, num_split
        input_2 = T.alloc_buffer((128, 16, 512, 4, 128)) # batch, head_num, q_len, num_split, ceil_div(kv_len, num_split)
        for batch in T.serial(128, annotations={"bind": "vblockIdx.3", "name": "batch"}):
            for head_num in T.serial(16, annotations={"bind": "vblockIdx.2", "name": "head_num"}):
                for q_len in T.serial(512, annotations={"bind": "vblockIdx.1", "name": "q_len"}):
                    for split in T.serial(4, annotations={"bind": "vblockIdx.0", "name": "split"}):
                        for kv_len in T.serial(128, annotations={"name": "kv_len", "tag": "fused"}):
                            for head_dim_qk in T.serial(64, annotations={"name": "head_dim_qk"}):
                                with T.block("prologue0"):
                                    v_batch, v_head_num, v_q_len, v_split, v_kv_len, v_head_dim_qk = T.axis.remap("SSSSSR", [batch, head_num, q_len, split, kv_len, head_dim_qk])
                                    T.reads(q[v_batch, v_head_num, v_q_len, v_head_dim_qk], k[v_batch, v_head_num, v_split * 128 + v_kv_len, v_head_dim_qk])
                                    T.writes(T_matmul_NT[v_batch, v_head_num, v_q_len, v_split, v_kv_len])
                                    with T.init():
                                        T_matmul_NT[v_batch, v_head_num, v_q_len, v_split, v_kv_len] = T.float32(0.0)
                                    T_matmul_NT[v_batch, v_head_num, v_q_len, v_split, v_kv_len] = T_matmul_NT[v_batch, v_head_num, v_q_len, v_split, v_kv_len] + T.Cast("float32", q[v_batch, v_head_num, v_q_len, v_head_dim_qk]) * T.Cast("float32", k[v_batch, v_head_num, v_split * 128 + v_kv_len, v_head_dim_qk])
        for batch in T.serial(128, annotations={"bind": "vblockIdx.3", "name": "batch"}):
            for head_num in T.serial(16, annotations={"bind": "vblockIdx.2", "name": "head_num"}):
                for q_len in T.serial(512, annotations={"bind": "vblockIdx.1", "name": "q_len"}):
                    for split in T.serial(4, annotations={"bind": "vblockIdx.0", "name": "split"}):
                        for kv_len in T.serial(128, annotations={"name": "kv_len", "tag": "fused"}):
                            with T.block("reduction0"):
                                v_batch, v_head_num, v_q_len, v_split, v_kv_len = T.axis.remap("SSSSR", [batch, head_num, q_len, split, kv_len])
                                T.reads(T_matmul_NT[v_batch, v_head_num, v_q_len, v_split, v_kv_len])
                                T.writes(T_softmax_maxelem[v_batch, v_head_num, v_q_len, v_split], prev_T_softmax_maxelem[v_batch, v_head_num, v_q_len, v_split], input_0[v_batch, v_head_num, v_q_len, v_split, v_kv_len])
                                with T.init():
                                    T_softmax_maxelem[v_batch, v_head_num, v_q_len, v_split] = T.float32("-inf")
                                prev_T_softmax_maxelem[v_batch, v_head_num, v_q_len, v_split] = T_softmax_maxelem[v_batch, v_head_num, v_q_len, v_split]
                                input_0[v_batch, v_head_num, v_q_len, v_split, v_kv_len] = T_matmul_NT[v_batch, v_head_num, v_q_len, v_split, v_kv_len] * (T.float32(1.0) / T.sqrt(T.float32(64.0)))
                                T_softmax_maxelem[v_batch, v_head_num, v_q_len, v_split] = T.max(T_softmax_maxelem[v_batch, v_head_num, v_q_len, v_split], input_0[v_batch, v_head_num, v_q_len, v_split, v_kv_len])
        for batch in T.serial(128, annotations={"bind": "vblockIdx.3", "name": "batch"}):
            for head_num in T.serial(16, annotations={"bind": "vblockIdx.2", "name": "head_num"}):
                for q_len in T.serial(512, annotations={"bind": "vblockIdx.1", "name": "q_len"}):
                    for split in T.serial(4, annotations={"bind": "vblockIdx.0", "name": "split"}):
                        for kv_len in T.serial(128, annotations={"name": "kv_len", "tag": "fused"}):
                            for head_dim_v in T.serial(64, annotations={"name": "head_dim_v"}):
                                with T.block("reduction1"):
                                    v_batch, v_head_num, v_q_len, v_split, v_head_dim_v, v_kv_len = T.axis.remap("SSSSSR", [batch, head_num, q_len, split, head_dim_v, kv_len])
                                    T.reads(T_softmax_maxelem[v_batch, v_head_num, v_q_len, v_split], prev_T_softmax_maxelem[v_batch, v_head_num, v_q_len, v_split], T_matmul_NT[v_batch, v_head_num, v_q_len, v_split, v_kv_len], v[v_batch, v_head_num, v_split * 128 + v_kv_len, v_head_dim_v])
                                    T.writes(T_matmul_NN[v_batch, v_head_num, v_q_len, v_split, v_head_dim_v], rescale_factor_1[v_batch, v_head_num, v_q_len, v_split], input_1[v_batch, v_head_num, v_q_len, v_split, v_kv_len])
                                    with T.init():
                                        T_matmul_NN[v_batch, v_head_num, v_q_len, v_split, v_head_dim_v] = T.float32(0.0)
                                    rescale_factor_1[v_batch, v_head_num, v_q_len, v_split] = T.exp(T.float32(-1.0) * T_softmax_maxelem[v_batch, v_head_num, v_q_len, v_split] + prev_T_softmax_maxelem[v_batch, v_head_num, v_q_len, v_split])
                                    input_1[v_batch, v_head_num, v_q_len, v_split, v_kv_len] = T.exp(T_matmul_NT[v_batch, v_head_num, v_q_len, v_split, v_kv_len] * (T.float32(1.0) / T.sqrt(T.float32(64.0))) - T_softmax_maxelem[v_batch, v_head_num, v_q_len, v_split])
                                    T_matmul_NN[v_batch, v_head_num, v_q_len, v_split, v_head_dim_v] = T_matmul_NN[v_batch, v_head_num, v_q_len, v_split, v_head_dim_v] * rescale_factor_1[v_batch, v_head_num, v_q_len, v_split]
                                    T_matmul_NN[v_batch, v_head_num, v_q_len, v_split, v_head_dim_v] = T_matmul_NN[v_batch, v_head_num, v_q_len, v_split, v_head_dim_v] + T.Cast("float32", T.Cast("float16", input_1[v_batch, v_head_num, v_q_len, v_split, v_kv_len])) * T.Cast("float32", v[v_batch, v_head_num, v_split * 128 + v_kv_len, v_head_dim_v])
        for batch in T.serial(128, annotations={"bind": "vblockIdx.3", "name": "batch"}):
            for head_num in T.serial(16, annotations={"bind": "vblockIdx.2", "name": "head_num"}):
                for q_len in T.serial(512, annotations={"bind": "vblockIdx.1", "name": "q_len"}):
                    for split in T.serial(4, annotations={"bind": "vblockIdx.0", "name": "split"}):
                        for kv_len in T.serial(128, annotations={"name": "kv_len", "tag": "fused"}):
                            with T.block("reduction2"):
                                v_batch, v_head_num, v_q_len, v_split, v_kv_len = T.axis.remap("SSSSR", [batch, head_num, q_len, split, kv_len])
                                T.reads(T_softmax_maxelem[v_batch, v_head_num, v_q_len, v_split], prev_T_softmax_maxelem[v_batch, v_head_num, v_q_len, v_split], T_matmul_NT[v_batch, v_head_num, v_q_len, v_split, v_kv_len])
                                T.writes(T_softmax_expsum[v_batch, v_head_num, v_q_len, v_split], rescale_factor_2[v_batch, v_head_num, v_q_len, v_split], input_2[v_batch, v_head_num, v_q_len, v_split, v_kv_len])
                                with T.init():
                                    T_softmax_expsum[v_batch, v_head_num, v_q_len, v_split] = T.float32(0.0)
                                rescale_factor_2[v_batch, v_head_num, v_q_len, v_split] = T.exp(T.float32(-1.0) * T_softmax_maxelem[v_batch, v_head_num, v_q_len, v_split] + prev_T_softmax_maxelem[v_batch, v_head_num, v_q_len, v_split])
                                input_2[v_batch, v_head_num, v_q_len, v_split, v_kv_len] = T.exp(T_matmul_NT[v_batch, v_head_num, v_q_len, v_split, v_kv_len] * (T.float32(1.0) / T.sqrt(T.float32(64.0))) - T_softmax_maxelem[v_batch, v_head_num, v_q_len, v_split])
                                T_softmax_expsum[v_batch, v_head_num, v_q_len, v_split] = T_softmax_expsum[v_batch, v_head_num, v_q_len, v_split] * rescale_factor_2[v_batch, v_head_num, v_q_len, v_split]
                                T_softmax_expsum[v_batch, v_head_num, v_q_len, v_split] = T_softmax_expsum[v_batch, v_head_num, v_q_len, v_split] + input_2[v_batch, v_head_num, v_q_len, v_split, v_kv_len]
        for batch in T.serial(128, annotations={"bind": "vblockIdx.3", "name": "batch"}):
            for head_num in T.serial(16, annotations={"bind": "vblockIdx.2", "name": "head_num"}):
                for q_len in T.serial(512, annotations={"bind": "vblockIdx.1", "name": "q_len"}):
                    for split in T.serial(4, annotations={"bind": "vblockIdx.0", "name": "split"}):
                        for head_dim_v in T.serial(64, annotations={"name": "head_dim_v"}):
                            with T.block("epilogue0"):
                                v_batch, v_head_num, v_q_len, v_split, v_head_dim_v = T.axis.remap("SSSSS", [batch, head_num, q_len, split, head_dim_v])
                                T.reads(T_matmul_NN[v_batch, v_head_num, v_q_len, v_split, v_head_dim_v], T_softmax_expsum[v_batch, v_head_num, v_q_len, v_split])
                                T.writes(part_output[v_batch, v_head_num, v_q_len, v_split, v_head_dim_v], part_max[v_batch, v_head_num, v_q_len, v_split], part_exp_sum[v_batch, v_head_num, v_q_len, v_split])
                                part_max[v_batch, v_head_num, v_q_len, v_split] = T.Cast("float16", T_softmax_maxelem[v_batch, v_head_num, v_q_len, v_split])
                                part_exp_sum[v_batch, v_head_num, v_q_len, v_split] = T.Cast("float16", T_softmax_expsum[v_batch, v_head_num, v_q_len, v_split])
                                # we put division in redfuser_flash_decoding2 for better performance
                                part_output[v_batch, v_head_num, v_q_len, v_split, v_head_dim_v] = T.Cast("float16", T_matmul_NN[v_batch, v_head_num, v_q_len, v_split, v_head_dim_v])

    @T.prim_func
    def redfuser_flash_decoding2(
        part_max: T.Buffer((128, 16, 512, 4), "float16"), # batch, head_num, q_len, num_split
        part_exp_sum: T.Buffer((128, 16, 512, 4), "float16"), # batch, head_num, q_len, num_split
        part_output: T.Buffer((128, 16, 512, 4, 64), "float16"), # batch, head_num, q_len, num_split, head_dim_v
        output: T.Buffer((128, 16, 512, 64), "float16")
    ):
        # 这里我们不对归约进行融合
        # We have d_k = reduce_xx( d_(k-1) * H(D_(k-1))^(-1) * H(D_k) ) in Eq.11
        # For max, H() = 1                                            => AllMax = reduce_max( Tmax_(k-1) )
        # For exp_sum, H(D_(k-1)) = exp(-Tmax), H(D_k) = exp(-AllMax) => AllExpSum = reduce_sum( Texpsum * exp(TMax - AllMax) )
        # For O, H(D_(k-1)) = exp(-Tmax), H(D_k) = exp(-AllMax)       => O = reduce_sum( Tsplit * exp(TMax - AllMax) )
        # And at last, we divide O by AllExpSum to get the final result
        T_all_max = T.alloc_buffer((128, 16, 512))
        T_all_exp_sum = T.alloc_buffer((128, 16, 512))
        T_all_o = T.alloc_buffer((128, 16, 512, 64))
        rescale_factor_1 = T.alloc_buffer((128, 16, 512, 4))
        input_1 = T.alloc_buffer((128, 16, 512, 4))
        rescale_factor_2 = T.alloc_buffer((128, 16, 512, 4))
        input_2 = T.alloc_buffer((128, 16, 512, 4, 64))
        for batch in T.serial(128, annotations={"bind": "vblockIdx.2", "name": "batch"}):
            for head_num in T.serial(16, annotations={"bind": "vblockIdx.1", "name": "head_num"}):
                for q_len in T.serial(512, annotations={"bind": "vblockIdx.0", "name": "q_len"}):
                    for split in T.serial(4, annotations={"name": "split"}):
                        with T.block("reduction0"):
                            v_batch, v_head_num, v_q_len, v_split = T.axis.remap("SSSR", [batch, head_num, q_len, split])
                            T.reads(part_max[v_batch, v_head_num, v_q_len, v_split])
                            T.writes(T_all_max[v_batch, v_head_num, v_q_len])
                            with T.init():
                                T_all_max[v_batch, v_head_num, v_q_len] = T.float32("-inf")
                            T_all_max[v_batch, v_head_num, v_q_len] = T.max(T_all_max[v_batch, v_head_num, v_q_len], T.Cast("float32", part_max[v_batch, v_head_num, v_q_len, v_split]))
        for batch in T.serial(128, annotations={"bind": "vblockIdx.2", "name": "batch"}):
            for head_num in T.serial(16, annotations={"bind": "vblockIdx.1", "name": "head_num"}):
                for q_len in T.serial(512, annotations={"bind": "vblockIdx.0", "name": "q_len"}):
                    for split in T.serial(4, annotations={"name": "split"}):
                        with T.block("reduction1"):
                            v_batch, v_head_num, v_q_len, v_split = T.axis.remap("SSSR", [batch, head_num, q_len, split])
                            T.reads(part_exp_sum[v_batch, v_head_num, v_q_len, v_split], part_max[v_batch, v_head_num, v_q_len, v_split], T_all_max[v_batch, v_head_num, v_q_len])
                            T.writes(T_all_exp_sum[v_batch, v_head_num, v_q_len], rescale_factor_1[v_batch, v_head_num, v_q_len, v_split], input_1[v_batch, v_head_num, v_q_len, v_split])
                            with T.init():
                                T_all_exp_sum[v_batch, v_head_num, v_q_len] = T.float32(0.0)
                            # cal rescale factor
                            rescale_factor_1[v_batch, v_head_num, v_q_len, v_split] = T.exp(T.Cast("float32", part_max[v_batch, v_head_num, v_q_len, v_split]) - T_all_max[v_batch, v_head_num, v_q_len])
                            # do rescale
                            input_1[v_batch, v_head_num, v_q_len, v_split] = T.Cast("float32", part_exp_sum[v_batch, v_head_num, v_q_len, v_split]) * rescale_factor_1[v_batch, v_head_num, v_q_len, v_split]
                            # do reduction
                            T_all_exp_sum[v_batch, v_head_num, v_q_len] = T_all_exp_sum[v_batch, v_head_num, v_q_len] + input_1[v_batch, v_head_num, v_q_len, v_split]
        for batch in T.serial(128, annotations={"bind": "vblockIdx.2", "name": "batch"}):
            for head_num in T.serial(16, annotations={"bind": "vblockIdx.1", "name": "head_num"}):
                for q_len in T.serial(512, annotations={"bind": "vblockIdx.0", "name": "q_len"}):
                    for split in T.serial(4, annotations={"name": "split"}):
                        for head_dim_v in T.serial(64, annotations={"name": "head_dim_v"}):
                            with T.block("reduction2"):
                                v_batch, v_head_num, v_q_len, v_head_dim_v, v_split = T.axis.remap("SSSSR", [batch, head_num, q_len, head_dim_v, split])
                                T.reads(part_output[v_batch, v_head_num, v_q_len, v_split, v_head_dim_v], part_max[v_batch, v_head_num, v_q_len, v_split], T_all_max[v_batch, v_head_num, v_q_len])
                                T.writes(T_all_o[v_batch, v_head_num, v_q_len, v_head_dim_v], rescale_factor_2[v_batch, v_head_num, v_q_len, v_split], input_2[v_batch, v_head_num, v_q_len, v_split, v_head_dim_v])
                                with T.init():
                                    T_all_o[v_batch, v_head_num, v_q_len, v_head_dim_v] = T.float32(0.0)
                                # cal rescale factor
                                rescale_factor_2[v_batch, v_head_num, v_q_len, v_split] = T.exp(T.Cast("float32", part_max[v_batch, v_head_num, v_q_len, v_split]) - T_all_max[v_batch, v_head_num, v_q_len])
                                # do rescale
                                input_2[v_batch, v_head_num, v_q_len, v_split, v_head_dim_v] = T.Cast("float32", part_output[v_batch, v_head_num, v_q_len, v_split, v_head_dim_v]) * rescale_factor_2[v_batch, v_head_num, v_q_len, v_split]
                                # do reduction
                                T_all_o[v_batch, v_head_num, v_q_len, v_head_dim_v] = T_all_o[v_batch, v_head_num, v_q_len, v_head_dim_v] + input_2[v_batch, v_head_num, v_q_len, v_split, v_head_dim_v]
        for batch in T.serial(128, annotations={"bind": "vblockIdx.2", "name": "batch"}):
            for head_num in T.serial(16, annotations={"bind": "vblockIdx.1", "name": "head_num"}):
                for q_len in T.serial(512, annotations={"bind": "vblockIdx.0", "name": "q_len"}):
                    for head_dim_v in T.serial(64, annotations={"name": "head_dim_v"}):
                        with T.block("epilogue0"):
                            v_batch, v_head_num, v_q_len, v_head_dim_v = T.axis.remap("SSSS", [batch, head_num, q_len, head_dim_v])
                            T.reads(T_all_o[v_batch, v_head_num, v_q_len, v_head_dim_v], T_all_exp_sum[v_batch, v_head_num, v_q_len])
                            T.writes(output[v_batch, v_head_num, v_q_len, v_head_dim_v])
                            output[v_batch, v_head_num, v_q_len, v_head_dim_v] = T.Cast("float16", T_all_o[v_batch, v_head_num, v_q_len, v_head_dim_v] / T_all_exp_sum[v_batch, v_head_num, v_q_len])


def test_decoding(tile_map):
    mod = Module

    passes = tvm.transform.Sequential(
        [
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

    import_stmt = "import tilelang\nimport tilelang.language as T\n\n"

    tilelang_prog = import_stmt + function_to_tilelang_script("redfuser_flash_decoding1", mod["redfuser_flash_decoding1"]) + function_to_tilelang_script("redfuser_flash_decoding2", mod["redfuser_flash_decoding2"])
    print(tilelang_prog, file=open(current_dir.joinpath("generated", f"generated_redfuser_flash_decoding.py"), "w"))
    print(f"Generated code saved to {current_dir.joinpath('generated', f'generated_redfuser_flash_decoding.py')}")

if __name__ == "__main__":
    tile_map = {"q_len": 64, "kv_len": 64, "head_dim_v": 64, "head_dim_qk": 64}
    test_decoding(tile_map)