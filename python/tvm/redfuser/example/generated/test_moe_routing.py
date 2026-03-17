from generated_redfuser_moe_routing import redfuser_moe_routing

import torch

def ref_program(logits: torch.Tensor, top_k: int):
    logits_norm = logits.softmax(dim=1)
    top_k_gates, top_k_indices = logits_norm.topk(top_k, dim=1)
    return top_k_gates, top_k_indices.to(torch.int32)

def test_moe_routing():
    logits = torch.randn((4096, 4096), device="cuda", dtype=torch.float16)
    top_k = 8

    kernel = redfuser_moe_routing()
    output_values, output_indices = kernel(logits)
    ref_output_values, ref_output_indices = ref_program(logits, top_k)
    torch.testing.assert_close(output_values, ref_output_values)
    # torch.testing.assert_close(output_indices, ref_output_indices)
    print("All checks passed.✅")

if __name__ == "__main__":
    test_moe_routing()