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