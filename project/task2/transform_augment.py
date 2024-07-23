import os
import cv2
import numpy as np
import random
import math

NEW_FOLDER = 'generated_augmented'
# Create new folder
if not os.path.exists(NEW_FOLDER):
    os.makedirs(NEW_FOLDER)

# Define transformations
def random_rotate(image):
    angle = random.choice([90, 180, 270])
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h))
    return rotated

def random_flip(image):
    flip_code = random.choice([-1, 0, 1])
    return cv2.flip(image, flip_code)

def random_brightness(image):
    value = random.randint(-50, 50)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v = cv2.add(v, value)
    v = np.clip(v, 0, 255)
    final_hsv = cv2.merge((h, s, v))
    brightened = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return brightened

# Resize images
images_directory = 'generated_resized'
for dirpath, dirnames, filenames in os.walk(images_directory):
    dir_name = dirpath.split('/')[-1]
    if dir_name == 'generated_resized':
        continue
    if int(dir_name) <= 2:
        print(f"Skipping folder {dir_name}.")
        continue
    num_photos = len(filenames)
    increasing_factor = int((math.sqrt(int(num_photos))/0.07)/num_photos)
    increasing_factor = 3
    print(f"Creating {increasing_factor * num_photos} new images from folder: {dir_name}")
    for filename in filenames:
        if filename.endswith('.jpg'):
            # Load image
            image_path = os.path.join(dirpath, filename)
            image = cv2.imread(image_path)
            if image is None:
                continue

            if not os.path.exists(os.path.join(NEW_FOLDER, dir_name)):
                os.makedirs(os.path.join(NEW_FOLDER, dir_name))

            # Keep original image
            cv2.imwrite(os.path.join(NEW_FOLDER, dir_name, filename), image)

            # Create 4 augmented images
            for i in range(increasing_factor):
                # Apply random transformations
                resized_image = image
                if random.choice([True, False]):
                    resized_image = random_rotate(resized_image)
                if random.choice([True, False]):
                    resized_image = random_flip(resized_image)
                if random.choice([True, False]):
                    resized_image = random_brightness(resized_image)

                # Save augmented image
                base_filename = os.path.splitext(filename)[0]
                cv2.imwrite(os.path.join(NEW_FOLDER, dir_name, f"{base_filename}_augmented_{i}.jpg"), resized_image)
            
print("Random data augmentation completed.")
