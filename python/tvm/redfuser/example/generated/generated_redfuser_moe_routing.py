import tilelang
import tilelang.language as T

@T.macro
def reduce_topk(
    input_frag: T.Buffer,
    topk_vals: T.Buffer,
    topk_indices: T.Buffer,
    topk: T.int32,
    axis: T.int32,
    start_offset: T.int32,
):
    blk_m = input_frag.shape[0]
    blk_n = input_frag.shape[1]

    expand_max_idx = T.alloc_fragment([blk_m, blk_n], T.int32)
    temp_max = T.alloc_fragment([blk_m], dtype=T.float32)
    temp_idx = T.alloc_fragment([blk_m], T.int32)

    curr_val = T.alloc_fragment([blk_m], dtype=T.float32)
    curr_idx = T.alloc_fragment([blk_m], T.int32)
    prev_val = T.alloc_fragment([blk_m], dtype=T.float32)
    prev_idx = T.alloc_fragment([blk_m], T.int32)

    for k in T.serial(topk):
        T.fill(expand_max_idx, start_offset + blk_n)
        T.reduce_max(input_frag, temp_max, dim=axis, clear=True)
        for i, j in T.Parallel(blk_m, blk_n):
            expand_max_idx[i, j] = T.if_then_else(
                temp_max[i] == input_frag[i, j],
                start_offset + j,
                expand_max_idx[i, j],
            )
        T.reduce_min(expand_max_idx, temp_idx, dim=axis, clear=True)
        for i, j in T.Parallel(blk_m, blk_n):
            input_frag[i, j] = T.if_then_else(
                temp_idx[i] == start_offset + j, -10000.0, input_frag[i, j]
            )
        for k_ in T.serial(topk - 1):
            pos = topk - k_ - 1
            for i in T.Parallel(blk_m):
                topk_indices[i, pos] = T.if_then_else(
                    pos == topk - 1 and temp_max[i] > topk_vals[i, pos],
                    temp_idx[i],
                    topk_indices[i, pos],
                )
                topk_vals[i, pos] = T.if_then_else(
                    pos == topk - 1 and temp_max[i] > topk_vals[i, pos],
                    temp_max[i],
                    topk_vals[i, pos],
                )

            for i in T.Parallel(blk_m):
                curr_val[i] = topk_vals[i, pos]
                curr_idx[i] = topk_indices[i, pos]
            for i in T.Parallel(blk_m):
                prev_val[i] = topk_vals[i, pos - 1]
                prev_idx[i] = topk_indices[i, pos - 1]
            
            for i in T.Parallel(blk_m):
                topk_indices[i, pos] = T.if_then_else(
                    curr_val[i] > prev_val[i],
                    prev_idx[i],
                    curr_idx[i],
                )
                topk_vals[i, pos] = T.if_then_else(
                    curr_val[i] > prev_val[i],
                    prev_val[i],
                    curr_val[i],
                )

            for i in T.Parallel(blk_m):
                topk_indices[i, pos - 1] = T.if_then_else(
                    curr_val[i] > prev_val[i],
                    curr_idx[i],
                    prev_idx[i],
                )
                topk_vals[i, pos - 1] = T.if_then_else(
                    curr_val[i] > prev_val[i],
                    curr_val[i],
                    prev_val[i],
                )

@tilelang.jit(out_idx=[1, 2])
def redfuser_moe_routing():
    @T.prim_func
    def kernel(A: T.Tensor([4096, 4096], "float16"), values: T.Tensor([4096, 8], "float16"), indices: T.Tensor([4096, 8], "int32")):
        with T.Kernel(32) as v_m_o:
            A_1 = T.alloc_fragment([128, 128], "float16")
            values_1 = T.alloc_fragment([128, 8], "float16")
            indices_1 = T.alloc_fragment([128, 8], "int32")
            topk_elem = T.alloc_fragment([128, 8], "float32")
            topk_indices = T.alloc_fragment([128, 8], "int32")
            softmax_maxelem = T.alloc_fragment([128], "float32")
            softmax_expsum = T.alloc_fragment([128], "float32")
            input_0_0 = T.alloc_fragment([128, 128], "float32")
            prev_softmax_maxelem = T.alloc_fragment([128], "float32")
            input_1_0 = T.alloc_fragment([128, 128], "float32")
            rescale_factor_2 = T.alloc_fragment([128], "float32")
            input_2_0 = T.alloc_fragment([128, 128], "float32")
            T.fill(topk_elem[0:128, 0:8], -100000.0)
            T.fill(topk_indices[0:128, 0:8], -1)
            T.fill(softmax_maxelem[0:128], -1000000.0)
            T.fill(softmax_expsum[0:128], 0.0)
            for v_k_o in T.Pipelined(0, 32, num_stages=1):
                T.copy(A[v_m_o * 128:v_m_o * 128 + 128, v_k_o * 128:v_k_o * 128 + 128], A_1[0:128, 0:128])
                for m_1, k_1 in T.Parallel(128, 128):
                    input_0_0[m_1, k_1] = T.Cast("float32", A_1[m_1, k_1])
                reduce_topk(input_0_0, topk_elem, topk_indices, 8, -1, v_k_o * 128)
                T.copy(softmax_maxelem[0:128], prev_softmax_maxelem[0:128])
                for m_1, k_1 in T.Parallel(128, 128):
                    input_1_0[m_1, k_1] = T.Cast("float32", A_1[m_1, k_1])
                T.reduce(input_1_0, softmax_maxelem, "max", 1, False)
                for m_1 in T.Parallel(128):
                    rescale_factor_2[m_1] = T.exp(-1.0 * softmax_maxelem[m_1] + prev_softmax_maxelem[m_1])
                for m_1, k_1 in T.Parallel(128, 128):
                    input_2_0[m_1, k_1] = T.exp(T.Cast("float32", A_1[m_1, k_1]) - softmax_maxelem[m_1])
                for m_1 in T.Parallel(128):
                    softmax_expsum[m_1] = softmax_expsum[m_1] * rescale_factor_2[m_1]
                T.reduce(input_2_0, softmax_expsum, "sum", 1, False)
            for v_topk_o in T.Pipelined(0, 8, num_stages=1):
                for m_1_1 in T.Parallel(128):
                    values_1[m_1_1, v_topk_o] = T.Cast("float16", T.exp(topk_elem[m_1_1, v_topk_o] - softmax_maxelem[m_1_1]) / softmax_expsum[m_1_1])
                T.copy(values_1[0:128, v_topk_o:v_topk_o + 1], values[v_m_o * 128:v_m_o * 128 + 128, v_topk_o:v_topk_o + 1])
            for v_topk_o_1 in T.Pipelined(0, 8, num_stages=1):
                T.copy(topk_indices[0:128, v_topk_o_1:v_topk_o_1 + 1], indices_1[0:128, v_topk_o_1:v_topk_o_1 + 1])
                T.copy(indices_1[0:128, v_topk_o_1:v_topk_o_1 + 1], indices[v_m_o * 128:v_m_o * 128 + 128, v_topk_o_1:v_topk_o_1 + 1])

    return kernel


