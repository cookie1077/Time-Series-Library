import torch
import random
fix_seed = 2021
random.seed(fix_seed)
torch.manual_seed(fix_seed)
print(torch.cuda.is_available())