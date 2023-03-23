# write me a python script that will split my dataset files into training files and testing files. Dataset consists of input image and ground truth image. Input images are in folder input_images and ground truth images are in folder ground_truth_images. Input images and ground truth images has the same filenames except the Input images has .jpg extension and ground truth images has .png extension. Please ignore any non image files in the dataset folders. Please store input images and ground truth images to separate folders. Please use a seed for the random sampling. All the folders are in the same folder as this script. Please clear all the folders before running the script. Please use these folder names test_gt/gt_all, test_imgs/imgs_all, train_gt/gt_all, train_imgs/imgs_all. Last thing, please write this whole prompt as a comment in the first line of the python script.

import os
import random
import shutil

# Set the seed for the random sampling
random.seed(42)

# Define the path to the dataset folder
path_to_dataset = "."

# Define the path to the input images folder
path_to_input_images = os.path.join(path_to_dataset, "input_images")

# Define the path to the ground truth images folder
path_to_ground_truth_images = os.path.join(path_to_dataset, "ground_truth_images")

# Define the path to the train images folder
path_to_train_images = os.path.join(path_to_dataset, "train_imgs", "imgs_all")

# Define the path to the train ground truth folder
path_to_train_ground_truth = os.path.join(path_to_dataset, "train_gt", "gt_all")

# Define the path to the test images folder
path_to_test_images = os.path.join(path_to_dataset, "test_imgs", "imgs_all")

# Define the path to the test ground truth folder
path_to_test_ground_truth = os.path.join(path_to_dataset, "test_gt", "gt_all")

# Create the train and test folders
os.makedirs(path_to_train_images, exist_ok=True)
os.makedirs(path_to_train_ground_truth, exist_ok=True)
os.makedirs(path_to_test_images, exist_ok=True)
os.makedirs(path_to_test_ground_truth, exist_ok=True)

# Get the list of input images
input_images = [f for f in os.listdir(path_to_input_images) if f.endswith(".jpg")]

# Shuffle the input images
random.shuffle(input_images)

# Split the input images into train and test sets
train_input_images = input_images[:int(0.8 * len(input_images))]
test_input_images = input_images[int(0.8 * len(input_images)):]

# Copy the train input images to the train images folder
for input_image in train_input_images:
    input_image_path = os.path.join(path_to_input_images, input_image)
    train_image_path = os.path.join(path_to_train_images, input_image)
    shutil.copy(input_image_path, train_image_path)

# Copy the test input images to the test images folder
for input_image in test_input_images:
    input_image_path = os.path.join(path_to_input_images, input_image)
    test_image_path = os.path.join(path_to_test_images, input_image)
    shutil.copy(input_image_path, test_image_path)

# Copy the train ground truth images to the train ground truth folder
for input_image in train_input_images:
    ground_truth_image = input_image.replace(".jpg", ".png")
    ground_truth_image_path = os.path.join(path_to_ground_truth_images, ground_truth_image)
    train_ground_truth_path = os.path.join(path_to_train_ground_truth, ground_truth_image)
    shutil.copy(ground_truth_image_path, train_ground_truth_path)

# Copy the test ground truth images to the test ground truth folder
for input_image in test_input_images:
    ground_truth_image = input_image.replace(".jpg", ".png")
    ground_truth_image_path = os.path.join(path_to_ground_truth_images, ground_truth_image)
    test_ground_truth_path = os.path.join(path_to_test_ground_truth, ground_truth_image)
    shutil.copy(ground_truth_image_path, test_ground_truth_path)