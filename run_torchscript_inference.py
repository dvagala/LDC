import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from torch.utils.mobile_optimizer import optimize_for_mobile
import cv2
import numpy as np
import os
import torch
from modelB4_scriptable import LDC

# image = cv2.imread('/Users/dvagala/School/LDC/data/35028.jpg', cv2.IMREAD_COLOR)
image = cv2.imread('/Users/dvagala/School/LDC/data/512.jpg', cv2.IMREAD_COLOR)

# image = cv2.resize(image, (512*2, 512*2))

image = torch.from_numpy(image.copy())
image = torch.permute(image, (2, 0, 1))
image = image.float()
image = image.unsqueeze(0)

print(f'image.shape: {image.shape}')

model = torch.jit.load('/Users/dvagala/School/LDC/checkpoints/BRIND/16/16_model_scripted.pt')

# model = LDC()
# model.load_state_dict(torch.load('/Users/dvagala/School/LDC/checkpoints/BRIND/16/16_model_scriptable.pt', map_location=torch.device('cpu')))

model.eval()


out = model(image)
out = out[len(out)-1]

# specific_value = torch.flatten(out)[0].item()
specific_value = torch.flatten(out)[125130].item()
print(specific_value)


out = out.squeeze()
out = torch.sigmoid(out)
out = torch.clamp(torch.mul(out, 255), min= 0, max= 255)
tensor = np.uint8(out.detach().numpy())
cv2.imwrite(os.path.join('/Users/dvagala/School/LDC/torchscript_inference', 'out.jpg'), tensor.astype(np.uint8))


# specific_value = out[0].tolist()[0][0][0][0]
# print(specific_value)
# assert specific_value == 0.07615365087985992
# print("ok")
