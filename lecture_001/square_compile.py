# TORCH_LOGS = "OUTPUT_CODE" python square_compile.py

import torch 

def square(a) :
    a = torch.square(a)
    return torch.square(a)

opt_square = torch.compile(square)
opt_square(torch.rndn(10000, 10000).cuda())