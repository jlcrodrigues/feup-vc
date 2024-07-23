import os
import cv2

NEW_FOLDER = 'generated_resized'
# create new  folder
if not os.path.exists(NEW_FOLDER):
    os.makedirs(NEW_FOLDER)

# Resize images
images_directory = 'generated'
for dirpath, dirnames, filenames in os.walk(images_directory):
    dir_name = dirpath.split('/')[-1]
    print("Transforming images from folder: ", dir_name)
    for filename in filenames:
        if filename.endswith('.jpg'):
            # Resize to 224x224
            img = cv2.imread(os.path.join(dirpath, filename))
            img = cv2.resize(img, (224, 224))
            if not os.path.exists(os.path.join(NEW_FOLDER, dir_name)):
                os.makedirs(os.path.join(NEW_FOLDER, dir_name))

            # Lossless compression
            cv2.imwrite(os.path.join(NEW_FOLDER, dir_name, filename), img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
