import torch
from filing_paths import path_model
import sys
sys.path.insert(1, path_model)
from model import gt_data
mask = torch.tensor([True,True,True,False,False,False])
for sequence in gt_data:
    print(sequence[:,0][mask])
