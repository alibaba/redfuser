import tilelang
import tilelang.language as T

@tilelang.jit(out_idx=[3, 4, 5])
def redfuser_flash_decoding1():
    @T.prim_func
    def kernel(q: T.Tensor([128, 16, 512, 64], "float16"), k: T.Tensor([128, 16, 512, 64], "float16"), v: T.Tensor([128, 16, 512, 64], "float16"), part_max: T.Tensor([128, 16, 512, 4], "float16"), part_exp_sum: T.Tensor([128, 16, 512, 4], "float16"), part_output: T.Tensor([128, 16, 512, 4, 64], "float16")):
        with T.Kernel(4, 8, 2048) as (v_split_o, v_q_len_o, fused_head_num_m):
            v_head_num_o, v_batch_o = T.index_to_coordinates(fused_head_num_m, [16, 128])
            q_1 = T.alloc_shared([64, 64], "float16")
            k_1 = T.alloc_shared([64, 64], "float16")
            v_1 = T.alloc_shared([64, 64], "float16")
            part_max_1 = T.alloc_fragment([64], "float16")
            part_exp_sum_1 = T.alloc_fragment([64], "float16")
            part_output_1 = T.alloc_fragment([64, 64], "float16")
            matmul_NT = T.alloc_fragment([64, 64], "float32")
            softmax_maxelem = T.alloc_fragment([64], "float32")
            matmul_NN = T.alloc_fragment([64, 64], "float32")
            softmax_expsum = T.alloc_fragment([64], "float32")
            prev_softmax_maxelem = T.alloc_fragment([64], "float32")
            input_0 = T.alloc_fragment([64, 64], "float32")
            rescale_factor_1 = T.alloc_fragment([64], "float32")
            input_1 = T.alloc_fragment([64, 64], "float32")
            rescale_factor_2 = T.alloc_fragment([64], "float32")
            input_2 = T.alloc_fragment([64, 64], "float32")
            input_1_fp16_0 = T.alloc_fragment([64, 64], "float16")
            T.copy(q[v_batch_o:v_batch_o + 1, v_head_num_o:v_head_num_o + 1, v_q_len_o * 64:v_q_len_o * 64 + 64, 0:64], q_1[0:64, 0:64])
            T.fill(softmax_maxelem[0:64], -T.infinity("float32"))
            T.fill(softmax_expsum[0:64], 0.0)
            T.fill(matmul_NN[0:64, 0:64], 0.0)
            for v_kv_len_o in T.Pipelined(0, 2, num_stages=1):
                T.fill(matmul_NT[0:64, 0:64], 0.0)
                T.copy(k[v_batch_o:v_batch_o + 1, v_head_num_o:v_head_num_o + 1, v_split_o * 128 + v_kv_len_o * 64:v_split_o * 128 + v_kv_len_o * 64 + 64, 0:64], k_1[0:64, 0:64])
                T.gemm(q_1, k_1, matmul_NT, transpose_B=True, policy=1)
                T.copy(softmax_maxelem[0:64], prev_softmax_maxelem[0:64])
                for q_len_1_m, kv_len_1_m in T.Parallel(64, 64):
                    input_0[q_len_1_m, kv_len_1_m] = matmul_NT[q_len_1_m, kv_len_1_m] * (1.0 / T.sqrt(64.0))
                T.reduce(input_0, softmax_maxelem, "max", 1, False)
                for q_len_1_m in T.Parallel(64):
                    rescale_factor_2[q_len_1_m] = T.exp(-1.0 * softmax_maxelem[q_len_1_m] + prev_softmax_maxelem[q_len_1_m])
                for q_len_1_m, kv_len_1_m in T.Parallel(64, 64):
                    input_2[q_len_1_m, kv_len_1_m] = T.exp(matmul_NT[q_len_1_m, kv_len_1_m] * (1.0 / T.sqrt(64.0)) - softmax_maxelem[q_len_1_m])
                for q_len_1_m in T.Parallel(64):
                    softmax_expsum[q_len_1_m] = softmax_expsum[q_len_1_m] * rescale_factor_2[q_len_1_m]
                T.reduce(input_2, softmax_expsum, "sum", 1, False)
                T.copy(v[v_batch_o:v_batch_o + 1, v_head_num_o:v_head_num_o + 1, v_split_o * 128 + v_kv_len_o * 64:v_split_o * 128 + v_kv_len_o * 64 + 64, 0:64], v_1[0:64, 0:64])
                for q_len_1 in T.Parallel(64):
                    rescale_factor_1[q_len_1] = T.exp(-1.0 * softmax_maxelem[q_len_1] + prev_softmax_maxelem[q_len_1])
                for q_len_1, kv_len_1 in T.Parallel(64, 64):
                    input_1[q_len_1, kv_len_1] = T.exp(matmul_NT[q_len_1, kv_len_1] * (1.0 / T.sqrt(64.0)) - softmax_maxelem[q_len_1])
                for q_len_1, head_dim_v_1 in T.Parallel(64, 64):
                    matmul_NN[q_len_1, head_dim_v_1] = matmul_NN[q_len_1, head_dim_v_1] * rescale_factor_1[q_len_1]
                T.copy(input_1[0:64, 0:64], input_1_fp16_0[0:64, 0:64])
                T.gemm(input_1_fp16_0, v_1, matmul_NN, policy=1)
            for q_len_1_1 in T.Parallel(64):
                part_max_1[q_len_1_1] = T.Cast("float16", softmax_maxelem[q_len_1_1])
            for q_len_1_1 in T.Parallel(64):
                part_exp_sum_1[q_len_1_1] = T.Cast("float16", softmax_expsum[q_len_1_1])
            for q_len_1_1, head_dim_v_1_1 in T.Parallel(64, 64):
                part_output_1[q_len_1_1, head_dim_v_1_1] = T.Cast("float16", matmul_NN[q_len_1_1, head_dim_v_1_1])
            T.copy(part_max_1[0:64], part_max[v_batch_o:v_batch_o + 1, v_head_num_o:v_head_num_o + 1, v_q_len_o * 64:v_q_len_o * 64 + 64, v_split_o:v_split_o + 1])
            T.copy(part_exp_sum_1[0:64], part_exp_sum[v_batch_o:v_batch_o + 1, v_head_num_o:v_head_num_o + 1, v_q_len_o * 64:v_q_len_o * 64 + 64, v_split_o:v_split_o + 1])
            T.copy(part_output_1[0:64, 0:64], part_output[v_batch_o:v_batch_o + 1, v_head_num_o:v_head_num_o + 1, v_q_len_o * 64:v_q_len_o * 64 + 64, v_split_o:v_split_o + 1, 0:64])

    return kernel

@tilelang.jit(out_idx=[3])
def redfuser_flash_decoding2():
    @T.prim_func
    def kernel(part_max: T.Tensor([128, 16, 512, 4], "float16"), part_exp_sum: T.Tensor([128, 16, 512, 4], "float16"), part_output: T.Tensor([128, 16, 512, 4, 64], "float16"), output: T.Tensor([128, 16, 512, 64], "float16")):
        with T.Kernel(8, 16, 128) as (v_q_len_o, v_head_num_o, v_batch_o):
            part_max_1 = T.alloc_fragment([64], "float16")
            part_exp_sum_1 = T.alloc_fragment([64], "float16")
            part_output_1 = T.alloc_fragment([64, 64], "float16")
            output_1 = T.alloc_fragment([64, 64], "float16")
            all_max = T.alloc_fragment([64], "float32")
            all_exp_sum = T.alloc_fragment([64], "float32")
            all_o = T.alloc_fragment([64, 64], "float32")
            rescale_factor_1 = T.alloc_fragment([64], "float32")
            input_1 = T.alloc_fragment([64], "float32")
            rescale_factor_2 = T.alloc_fragment([64], "float32")
            input_2 = T.alloc_fragment([64, 64], "float32")
            T.fill(all_max[0:64], -T.infinity("float32"))
            for v_split_o in T.Pipelined(0, 4, num_stages=1):
                T.copy(part_max[v_batch_o:v_batch_o + 1, v_head_num_o:v_head_num_o + 1, v_q_len_o * 64:v_q_len_o * 64 + 64, v_split_o:v_split_o + 1], part_max_1[0:64])
                for q_len_1 in T.Parallel(64):
                    all_max[q_len_1] = T.max(all_max[q_len_1], T.Cast("float32", part_max_1[q_len_1]))
            T.fill(all_exp_sum[0:64], 0.0)
            for v_split_o_1 in T.Pipelined(0, 4, num_stages=1):
                T.copy(part_max[v_batch_o:v_batch_o + 1, v_head_num_o:v_head_num_o + 1, v_q_len_o * 64:v_q_len_o * 64 + 64, v_split_o_1:v_split_o_1 + 1], part_max_1[0:64])
                T.copy(part_exp_sum[v_batch_o:v_batch_o + 1, v_head_num_o:v_head_num_o + 1, v_q_len_o * 64:v_q_len_o * 64 + 64, v_split_o_1:v_split_o_1 + 1], part_exp_sum_1[0:64])
                for q_len_1_1 in T.Parallel(64):
                    rescale_factor_1[q_len_1_1] = T.exp(T.Cast("float32", part_max_1[q_len_1_1]) - all_max[q_len_1_1])
                for q_len_1_1 in T.Parallel(64):
                    input_1[q_len_1_1] = T.Cast("float32", part_exp_sum_1[q_len_1_1]) * rescale_factor_1[q_len_1_1]
                for q_len_1_1 in T.Parallel(64):
                    all_exp_sum[q_len_1_1] = all_exp_sum[q_len_1_1] + input_1[q_len_1_1]
            T.fill(all_o[0:64, 0:64], 0.0)
            for v_split_o_2 in T.Pipelined(0, 4, num_stages=1):
                T.copy(part_max[v_batch_o:v_batch_o + 1, v_head_num_o:v_head_num_o + 1, v_q_len_o * 64:v_q_len_o * 64 + 64, v_split_o_2:v_split_o_2 + 1], part_max_1[0:64])
                T.copy(part_output[v_batch_o:v_batch_o + 1, v_head_num_o:v_head_num_o + 1, v_q_len_o * 64:v_q_len_o * 64 + 64, v_split_o_2:v_split_o_2 + 1, 0:64], part_output_1[0:64, 0:64])
                for q_len_1_2 in T.Parallel(64):
                    rescale_factor_2[q_len_1_2] = T.exp(T.Cast("float32", part_max_1[q_len_1_2]) - all_max[q_len_1_2])
                for q_len_1_2, head_dim_v_1 in T.Parallel(64, 64):
                    input_2[q_len_1_2, head_dim_v_1] = T.Cast("float32", part_output_1[q_len_1_2, head_dim_v_1]) * rescale_factor_2[q_len_1_2]
                for q_len_1_2, head_dim_v_1 in T.Parallel(64, 64):
                    all_o[q_len_1_2, head_dim_v_1] = all_o[q_len_1_2, head_dim_v_1] + input_2[q_len_1_2, head_dim_v_1]
            for q_len_1_3, head_dim_v_1_1 in T.Parallel(64, 64):
                output_1[q_len_1_3, head_dim_v_1_1] = T.Cast("float16", all_o[q_len_1_3, head_dim_v_1_1] / all_exp_sum[q_len_1_3])
            T.copy(output_1[0:64, 0:64], output[v_batch_o:v_batch_o + 1, v_head_num_o:v_head_num_o + 1, v_q_len_o * 64:v_q_len_o * 64 + 64, 0:64])

    return kernel


