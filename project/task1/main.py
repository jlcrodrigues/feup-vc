import json
from dataclasses import dataclass, asdict
from typing import List
import cv2
import numpy as np
import sys
import os

DEBUG = False
OUTPUT_PATH = 'output.json'
RESIZE_SHAPE = (500, 500)


@dataclass
class DetectedObject:
    xmin: int
    ymin: int
    xmax: int
    ymax: int


@dataclass
class ImageResult:
    file_name: str
    num_colors: int
    num_detections: int
    detected_objects: List[DetectedObject]


@dataclass
class Results:
    results: List[ImageResult]


def load_input(input_path: str) -> List[str]:
    with open(input_path) as f:
        data = json.load(f)
    return data["image_files"]


def store_output(output_path: str, results: Results) -> None:
    with open(output_path, 'w') as f:
        json.dump(asdict(results), f)


def display_results(result: ImageResult, img, contours):
    print("=" * 50)
    print("Displaying results for: " + result.file_name)
    print("Number of objects detected:   " + str(result.num_detections))
    print("Number of colors detected:    " + str(result.num_colors))

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow('Image with contours', img)
    cv2.imshow('Original Image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def process_images(samples: List[str], samples_dir, debug=False) -> Results:
    results = []
    for sample in samples:
        results.append(process_image(
            os.path.join(samples_dir, sample), debug=debug))
    return Results(results)


def process_image(sample_path, debug=False) -> ImageResult:
    """
    Image processing pipeline.
    As per described in the report:
        - Preprocess image (resize, blur)
        - Find contours (canny, dilate, findContours)
        - Get colors (mean color of each contour)
    """
    if not os.path.exists(sample_path):
        raise FileNotFoundError(
            f"File not found: {sample_path}, check samples_dir.")
    img = cv2.imread(sample_path)
    resized_img = preprocess_image(img, resize_shape=RESIZE_SHAPE, blur=False)
    preprocessed_img = preprocess_image(resized_img)
    contours = get_contours(preprocessed_img)
    num_colors = get_colors(preprocessed_img, contours)

    result = ImageResult(
        file_name=sample_path,
        num_colors=num_colors,
        num_detections=len(contours),
        detected_objects=get_detected_objects(img, preprocessed_img, contours)
    )

    if debug:
        display_results(result, resized_img, contours)

    return result


def main():
    """
    Main function to run the program.
    The program expects two arguments:
    - input_path: Path to the input JSON file
    - samples_dir: (optional) Directory containing the sample images
    """
    if len(sys.argv) < 2:
        print("Usage: python main.py <input_file> [<samples_dir>]")
        print()
        print("Example:")
        print("  python main.py input.json samples")
        sys.exit(1)
    input_path = sys.argv[1]
    samples_dir = sys.argv[2] if len(sys.argv) > 2 else '.'

    sample_paths = load_input(input_path)
    result = process_images(sample_paths, samples_dir, debug=DEBUG)
    store_output(OUTPUT_PATH, result)


###################### Image processing functions ######################

def preprocess_image(img, grayscale=False, resize_shape=None, blur=True):
    """
    Preprocess image. This applies the following (some optional) steps:
    - Convert image to grayscale
    - Apply median blur to reduce noise

    Args:
        img: Image to preprocess
        grayscale: Convert image to grayscale if True
        resize_shape: Resize image to this shape if not None
        blur: If true, applies a median blur to the image

    Returns:
        Preprocessed image
    """
    median_blur_ksize = 13

    # Resize
    if resize_shape is not None:
        img = cv2.resize(img, resize_shape, interpolation=cv2.INTER_AREA)

    # Convert image to grayscale
    if grayscale:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Blur
    if blur:
        img = cv2.medianBlur(img, median_blur_ksize)

    return img


def get_contours(img) -> List:
    mask = cv2.Canny(img, 40, 210)
    dilated = cv2.dilate(mask, None, iterations=1)
    contours, hierarchy = cv2.findContours(
        dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # Filter overlapping contours
    contours_to_remove = []
    for i, cnt in enumerate(contours):
        for j, cnt2 in enumerate(contours):
            if i != j:
                x, y, w, h = cv2.boundingRect(cnt2)
                x2, y2, w2, h2 = cv2.boundingRect(cnt)
                if (x <= x2) and ((x + w) >= (x2 + w2)) and (y <= y2) and ((y + h) >= (y2 + h2)):
                    contours_to_remove.append(i)

    # Removing contours that touch the image corners
    for i, cnt in enumerate(contours):
        x, y, w, h = cv2.boundingRect(cnt)
        if (x == 0 and y == 0) or (x == 0 and y + h == img.shape[0]) or (x + w == img.shape[1] and y == 0) or (x + w == img.shape[1] and y + h == img.shape[0]):
            contours_to_remove.append(i)

    contours = [cnt for i, cnt in enumerate(
        contours) if i not in contours_to_remove]

    return contours


def get_colors(img, contours):
    average_colors = []
    for contour in contours:
        mask = np.zeros(img.shape[:2], np.uint8)
        cv2.drawContours(mask, [contour], -1, (255), -1)

        mean_color = find_dominant_color(img, mask)

        if not is_color_similar(mean_color, average_colors, color_space="lab"):
            average_colors.append(mean_color)
        mask.fill(0)

    return len(average_colors)


def find_dominant_color(image, mask=None):
    cv2.cvtColor(image, cv2.COLOR_BGR2LAB, dst=image)
    cv2.bitwise_and(image, image, mask=mask, dst=image)
    cv2.medianBlur(image, 13, dst=image)
    return cv2.mean(image, mask=mask)[:3]


def is_color_similar(color, color_list, color_space="rgb"):
    """Check if the given color is similar to any color in the list."""
    if color_space == "rgb":
        RGB_THRESHOLD = 0.022
        max_distance = np.sqrt((0.3+0.59+0.11) * 255**2)
        for existing_color in color_list:
            color1_np = np.array(color)
            color2_np = np.array(existing_color)
            color_distance = np.sqrt(
                color1_np[0] * 0.3 + color1_np[1] * 0.59 + color1_np[2] * 0.11)

            similarity = (max_distance - color_distance) / max_distance
            if similarity > 1 - RGB_THRESHOLD:
                return True
        return False
    elif color_space == "lab":
        LAB_THRESHOLD = 0.00850
        max_distance = np.sqrt(3 * 255**2)
        for existing_color in color_list:
            color1_np = np.array(color)
            color2_np = np.array(existing_color)
            color_distance = np.sqrt(np.sum((color1_np - color2_np)**2))
            similarity = (max_distance - color_distance) / max_distance
            if similarity > 1 - LAB_THRESHOLD:
                return True
        return False
    elif color_space == "hsv":
        HSV_THRESHOLD = 0.05
        for existing_color in color_list:
            hue_distance = min(
                abs(color[0] - existing_color[0]), 1 - abs(color[0] - existing_color[0]))
            saturation_distance = abs(color[1] - existing_color[1])
            value_distance = abs(color[2] - existing_color[2])
            color_distance = np.sqrt(
                hue_distance**2 + saturation_distance**2 + value_distance**2)
            similarity = (1 - color_distance)
            if similarity > 1 - HSV_THRESHOLD:
                return True
        return False
    else:
        raise ValueError("Invalid color space.")


def get_detected_objects(img, preprocessed_img, contours):
    """
    Get the deteced objects bounding boxes from the contours.
    """
    detected_objects = [DetectedObject(
        x, y, x+w, y+h) for x, y, w, h in [cv2.boundingRect(cnt) for cnt in contours]]
    scale_factor = img.shape[0] / preprocessed_img.shape[0]
    detected_objects = [DetectedObject(
        int(obj.xmin * scale_factor),
        int(obj.ymin * scale_factor),
        int(obj.xmax * scale_factor),
        int(obj.ymax * scale_factor)
    ) for obj in detected_objects]
    return detected_objects


if __name__ == '__main__':
    main()
