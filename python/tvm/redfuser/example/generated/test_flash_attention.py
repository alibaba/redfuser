from generated_redfuser_flash_attention import redfuser_flash_attention

import torch
import torch.nn.functional as F

def ref_program(q, k, v):
    return F.scaled_dot_product_attention(q, k, v)

def test_flash_attention():
    q = torch.randn((128, 16, 512, 64), device="cuda", dtype=torch.float16)
    k = torch.randn((128, 16, 512, 64), device="cuda", dtype=torch.float16)
    v = torch.randn((128, 16, 512, 64), device="cuda", dtype=torch.float16)

    kernel = redfuser_flash_attention()
    output = kernel(q, k, v)
    ref_output = ref_program(q, k, v)
    torch.testing.assert_close(output, ref_output, rtol=1e-2, atol=1e-2)
    print("All checks passed.✅")

if __name__ == "__main__":
    test_flash_attention()