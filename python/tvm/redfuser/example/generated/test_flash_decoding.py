from generated_redfuser_flash_decoding import redfuser_flash_decoding1, redfuser_flash_decoding2

import torch
import torch.nn.functional as F

def ref_program(q, k, v):
    return F.scaled_dot_product_attention(q, k, v)

def test_flash_decoding():
    q = torch.randn((128, 16, 512, 64), device="cuda", dtype=torch.float16)
    k = torch.randn((128, 16, 512, 64), device="cuda", dtype=torch.float16)
    v = torch.randn((128, 16, 512, 64), device="cuda", dtype=torch.float16)

    kernel1 = redfuser_flash_decoding1()
    kernel2 = redfuser_flash_decoding2()
    part_max, part_exp_sum, part_output = kernel1(q, k, v)
    output = kernel2(part_max, part_exp_sum, part_output)
    ref_output = ref_program(q, k, v)
    torch.testing.assert_close(output, ref_output, rtol=1e-2, atol=1e-2)
    print("All checks passed.✅")

if __name__ == "__main__":
    test_flash_decoding()