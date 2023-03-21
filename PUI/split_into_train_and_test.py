# write me a python script that will split my dataset files into training files and testing files. Dataset consists of input image and ground truth image. Input images are in folder input_images and ground truth images are in folder ground_truth_images. Input images and ground truth images has the same filenames except the Input images has .jpg extension and ground truth images has .png extension. Please ignore any non image files in the dataset folders. Please store input images and ground truth images to separate folders. Please use a seed for the random sampling. All the folders are in the same folder as this script. Please clear all the folders before running the script. Please use these folder names test_gt, test_imgs, train_gt, train_imgs.

import os
import random
import shutil

def split_dataset(input_dir, gt_dir, output_dir, split_ratio, seed_value):
    random.seed(seed_value)
    input_images = os.listdir(os.path.join(os.path.dirname(__file__), input_dir))
    gt_images = os.listdir(os.path.join(os.path.dirname(__file__), gt_dir))
    images = [f for f in input_images if f.endswith('.jpg') and os.path.isfile(os.path.join(os.path.join(os.path.dirname(__file__), gt_dir), f[:-4] + '.png'))]
    num_images = len(images)
    num_train = int(num_images * split_ratio)
    train_images = random.sample(images, num_train)
    test_images = list(set(images) - set(train_images))
    train_input_dir = os.path.join(os.path.join(os.path.dirname(__file__), output_dir), 'train_imgs')
    train_gt_dir = os.path.join(os.path.join(os.path.dirname(__file__), output_dir), 'train_gt')
    test_input_dir = os.path.join(os.path.join(os.path.dirname(__file__), output_dir), 'test_imgs')
    test_gt_dir = os.path.join(os.path.join(os.path.dirname(__file__), output_dir), 'test_gt')
    shutil.rmtree(os.path.join(os.path.dirname(__file__), output_dir), ignore_errors=True)
    os.makedirs(train_input_dir, exist_ok=True)
    os.makedirs(train_gt_dir, exist_ok=True)
    os.makedirs(test_input_dir, exist_ok=True)
    os.makedirs(test_gt_dir, exist_ok=True)
    for image in train_images:
        input_image_path = os.path.join(os.path.join(os.path.dirname(__file__), input_dir), image)
        gt_image_path = os.path.join(os.path.join(os.path.dirname(__file__), gt_dir), image[:-4] + '.png')
        shutil.copy(input_image_path, train_input_dir)
        shutil.copy(gt_image_path, train_gt_dir)
    for image in test_images:
        input_image_path = os.path.join(os.path.join(os.path.dirname(__file__), input_dir), image)
        gt_image_path = os.path.join(os.path.join(os.path.dirname(__file__), gt_dir), image[:-4] + '.png')
        shutil.copy(input_image_path, test_input_dir)
        shutil.copy(gt_image_path, test_gt_dir)

input_dir = 'input_images'
gt_dir = 'ground_truth_images'
output_dir = 'output'
split_ratio = 0.8
seed_value = 42
split_dataset(input_dir, gt_dir, output_dir, split_ratio, seed_value)