# # Hey, write me a python script for image dataset augmentation. Dataset consists of jpg input images that are in input_images directory and png ground truth images that are in ground_truth_images directory. Please only take images from these directories, ignore any other files. The augmented dataset will be stored in input_images_augmented and ground_truth_images_augmented. When doing image augmentation you can use these transformations: vertical flip, horizontal flip, rotation by multiple angles, brightness change, hue rotation, saturation change. You can combine these transformation in any way you want. The augmented dataset should be 10x larger as the original dataset.


from PIL import Image, ImageEnhance
import os
import random
import math
import cv2
import numpy

random.seed(10)

input_dir = 'input_images'
ground_truth_dir = 'ground_truth_images'
output_dir = 'input_images_augmented'
ground_truth_output_dir = 'ground_truth_images_augmented'

def largest_rotated_rect(w, h, angle):
    """
    Given a rectangle of size wxh that has been rotated by 'angle' (in
    radians), computes the width and height of the largest possible
    axis-aligned rectangle within the rotated rectangle.

    Original JS code by 'Andri' and Magnus Hoff from Stack Overflow

    Converted to Python by Aaron Snoswell
    """

    quadrant = int(math.floor(angle / (math.pi / 2))) & 3
    sign_alpha = angle if ((quadrant & 1) == 0) else math.pi - angle
    alpha = (sign_alpha % math.pi + math.pi) % math.pi

    bb_w = w * math.cos(alpha) + h * math.sin(alpha)
    bb_h = w * math.sin(alpha) + h * math.cos(alpha)

    gamma = math.atan2(bb_w, bb_w) if (w < h) else math.atan2(bb_w, bb_w)

    delta = math.pi - alpha - gamma

    length = h if (w < h) else w

    d = length * math.cos(alpha)
    a = d * math.sin(alpha) / math.sin(delta)

    y = a * math.cos(gamma)
    x = y * math.tan(gamma)

    return (
        bb_w - 2 * x,
        bb_h - 2 * y
    )


def crop_around_center(image, width, height):
    """
    Given a NumPy / OpenCV 2 image, crops it to the given width and height,
    around it's centre point
    """

    img_width, img_height = image.size
    image_center = (int(img_width * 0.5), int(img_height * 0.5))

    if(width > img_width):
        width = img_width

    if(height > img_height):
        height = img_height

    x1 = int(image_center[0] - width * 0.48)
    x2 = int(image_center[0] + width * 0.48)
    y1 = int(image_center[1] - height * 0.48)
    y2 = int(image_center[1] + height * 0.48)

    return image.crop((x1, y1, x2, y2))

def rotate_without_black_borders(img, angle):
    img = img.rotate(angle)
    img = crop_around_center(
        img,
        *largest_rotated_rect(width, height, math.radians(angle))
    )
    return img


if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for filename in os.listdir(output_dir):
    file_path = os.path.join(output_dir, filename)
    if os.path.isfile(file_path) or os.path.islink(file_path):
        os.unlink(file_path)
    elif os.path.isdir(file_path):
        os.rmdir(file_path)

if not os.path.exists(ground_truth_output_dir):
    os.makedirs(ground_truth_output_dir)
for filename in os.listdir(ground_truth_output_dir):
    file_path = os.path.join(ground_truth_output_dir, filename)
    if os.path.isfile(file_path) or os.path.islink(file_path):
        os.unlink(file_path)
    elif os.path.isdir(file_path):
        os.rmdir(file_path)

for filename in os.listdir(input_dir):
    if filename.endswith('.jpg') or filename.endswith('.jpeg'):
        input_path = os.path.join(input_dir, filename)
        ground_truth_path = os.path.join(ground_truth_dir, f'{filename[:-4]}.png')
        with Image.open(input_path) as original_img, Image.open(ground_truth_path) as original_gt_img:
            width, height = original_img.size
            for i in range(20):
                img = original_img
                gt_img = original_gt_img
                # Vertical flip
                if random.random() < 0.5:
                    img = img.transpose(Image.FLIP_TOP_BOTTOM)
                    gt_img = gt_img.transpose(Image.FLIP_TOP_BOTTOM)

                if random.random() < 0.5:
                    img = img.transpose(Image.FLIP_LEFT_RIGHT)
                    gt_img = gt_img.transpose(Image.FLIP_LEFT_RIGHT)

                # Rotation by multiple angles
                angle = random.randint(0, 360)
                img = rotate_without_black_borders(img, angle)
                gt_img = rotate_without_black_borders(gt_img, angle)

                opencvImage = cv2.cvtColor(numpy.array(img), cv2.COLOR_RGB2HSV)
                h,s,v = cv2.split(opencvImage)
                hue_new = numpy.clip(h + random.uniform(-180, 180), 0, 180).astype(numpy.uint8)
                sat_new = numpy.clip(s + random.uniform(-15, 15), 0, 255).astype(numpy.uint8)
                val_new = numpy.clip(v + random.uniform(-50, 50), 0, 255).astype(numpy.uint8)
                bgr_new = cv2.cvtColor(cv2.merge([hue_new, sat_new, val_new]), cv2.COLOR_HSV2BGR)
                img = Image.fromarray(bgr_new)

                output_path = os.path.join(output_dir, f'{filename[:-4]}_{i}.jpg')
                ground_truth_output_path = os.path.join(ground_truth_output_dir, f'{filename[:-4]}_{i}.png')

                img.save(output_path)
                gt_img.save(ground_truth_output_path)
