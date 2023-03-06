import argparse
import os
import time, platform

import cv2
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from thop import profile
import torchvision
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from torch.utils.mobile_optimizer import optimize_for_mobile


from dataset import DATASET_NAMES, BipedDataset, TestDataset, dataset_info
# from loss import *
from loss2 import *
# from modelB6 import LDC
# from modelB5 import LDC
from modelB4_scriptable import LDC
# from modelB4 import LDC
# from modelB3 import LDC
# from modelB2 import LDC
# from model6 import LDC


trained_model_file = sys.argv[1]



device = torch.device('cpu' if torch.cuda.device_count() == 0
			else 'cuda')

# Instantiate model and move it to the computing device
model = LDC().to(device)



model.load_state_dict(torch.load(trained_model_file,
				map_location=device))

# quantized_model = torch.quantization.convert(model)
scripted_model = torch.jit.script(model)
# opt_model = optimize_for_mobile(scripted_model)
scripted_model.save(f'converted_{trained_model_file}')

# model.eval()
