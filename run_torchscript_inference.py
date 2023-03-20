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
from os import listdir
from os.path import isfile, join




dir_path = '/Users/dvagala/School/LDC/data'

input_images = [join(dir_path, f) for f in listdir(dir_path) if isfile(join(dir_path, f)) and ('.jpg' in f or '.png' in f)]

# input_images = ['/Users/dvagala/School/LDC/data/0a0b61f0-9e6f-11ed-ad17-9f0b773a453f_orig-photo.jpg']


# model = torch.jit.load('/Users/dvagala/School/LDC/checkpoints/BRIND/16/16_model_scripted.pt')
# model = torch.jit.load('/Users/dvagala/School/LDC/checkpoints/B3_16_model_scripted.pt')
# model = torch.jit.load('/Users/dvagala/School/LDC/checkpoints/B4_ditched_previous_layers_16_model_scripted.pt')
model = torch.jit.load('/Users/dvagala/School/LDC/checkpoints/B4_16_model_scripted.pt')
# model = torch.jit.load('/Users/dvagala/School/LDC/checkpoints/on_biped_augmented_with_thresholding_17_model_scripted.pt')
# model = torch.jit.load('/Users/dvagala/School/LDC/checkpoints/trained_on_BIPED_without_augmentation_with_thresholding_17_epoch_scripted.pt')
# model = torch.jit.load('/Users/dvagala/School/LDC/checkpoints/trained_on_BIPED_without_augmentation_without_thresholding_17_epoch_scripted.pt')

# model = LDC()
# model.load_state_dict(torch.load('/Users/dvagala/School/LDC/checkpoints/BRIND/16/16_model_scriptable.pt', map_location=torch.device('cpu')))
model.eval()

for image_path in input_images:
	print(f'image_path: {image_path}')

	# image = cv2.imread('/Users/dvagala/School/LDC/data/35028.jpg', cv2.IMREAD_COLOR)
	# image = cv2.imread('/Users/dvagala/School/LDC/data/512.jpg', cv2.IMREAD_COLOR)
	image = cv2.imread(image_path, cv2.IMREAD_COLOR)

	orig_height = image.shape[0]
	orig_width = image.shape[1]
    
	image = cv2.resize(image, (512*2, 512*2))

	image = torch.from_numpy(image.copy())
	image = torch.permute(image, (2, 0, 1))
	image = image.float()
	image = image.unsqueeze(0)

	print(f'image.shape: {image.shape}')


	out = model(image)
	out = out[len(out)-1]

	print(f'specific value [125130]: {torch.flatten(out)[125130].item()}')


	out = out.squeeze()
	out = torch.sigmoid(out)
	out = torch.clamp(torch.mul(out, 255), min= 0, max= 255)
	tensor = np.uint8(out.detach().numpy())
	image_file_name = image_path.split('/')[-1]

	final_image = cv2.resize(tensor.astype(np.uint8), (orig_width, orig_height))

	# cv2.imwrite(os.path.join('/Users/dvagala/School/LDC/torchscript_inference', 'B3_out.jpg'), tensor.astype(np.uint8))
	cv2.imwrite(os.path.join('/Users/dvagala/School/LDC/torchscript_inference',image_file_name),final_image )
	# cv2.imwrite(os.path.join('/Users/dvagala/School/LDC/torchscript_inference', 'B4_dithed_out.jpg'), tensor.astype(np.uint8))

