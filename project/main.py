import json
from dataclasses import dataclass, asdict
from typing import List
import cv2
import numpy as np

DEBUG = True

INPUT_PATH = 'demo.json'
#INPUT_PATH = 'test/input/full.json'
OUTPUT_PATH = 'output.json'
SAMPLES_PATH = 'samples/'


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


def run_pipeline(samples: List[str], debug=False) -> Results:
    results = []
    for sample in samples:
        results.append(process_image(sample, debug=debug))
    return Results(results)


def process_image(sample_path, debug=False) -> ImageResult:
    img = cv2.imread(SAMPLES_PATH + sample_path)
    resized_img = preprocess_image(img, resize_shape=(500, 500), blur=False)
    preprocessed_img = preprocess_image(resized_img)
    cv2.imshow('Original Image', img)
    contours = get_contours(preprocessed_img)
    num_colors = get_colors(preprocessed_img, contours)

    result = ImageResult(
        file_name=sample_path,
        num_colors=num_colors,
        num_detections=len(contours),
        detected_objects=[]
    )

    if debug:
        display_results(result, resized_img, contours)

    return result


def main():
    sample_paths = load_input(INPUT_PATH)
    result = run_pipeline(sample_paths, debug=DEBUG)
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

    Returns:
        Preprocessed image
    """

    # Adjustable parameters
    median_blur_ksize = 13
    gaussian_blur_ksize = 7

    # Resize
    if resize_shape is not None:
        img = cv2.resize(img, resize_shape, interpolation=cv2.INTER_AREA)

    # Convert image to grayscale
    if grayscale:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if blur:
        # Median blur
        img = cv2.medianBlur(img, median_blur_ksize)

        # gaussian blur
        #img = cv2.GaussianBlur(
         #   img, (gaussian_blur_ksize, gaussian_blur_ksize), 0)

    return img

def intersection_area(cnt1, cnt2, img):
    """Find the intersection area between two contours."""
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    blank1 = np.zeros_like(gray_img)
    blank2 = np.zeros_like(gray_img)

    cv2.drawContours(blank1, [cnt1], 0, (255), thickness=cv2.FILLED)
    cv2.drawContours(blank2, [cnt2], 0, (255), thickness=cv2.FILLED)

    intersection = cv2.bitwise_and(blank1, blank2)

    contours, _ = cv2.findContours(
        intersection, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    intersection_area = 0
    for contour in contours:
        intersection_area += cv2.contourArea(contour)
    return intersection_area


def get_contours(img) -> List:
    # Apply threshold
    # tresh = cv2.threshold(gray_img, 100, 255, cv2.THRESH_BINARY)[1]
    tresh = img

    # TODO: see if this is worth it
    # # apply erosions to reduce size of foreground objects
    # mask = tresh.copy()
    # mask = cv2.erode(mask, None, iterations=5)
    # cv2.imshow('Eroded Image', mask)
    # cv2.waitKey(0)
    # cv2.destroyWindow('Eroded Image')

    # Canny edge filter
    mask = cv2.Canny(tresh, 40, 210)

    dilated = cv2.dilate(mask, None, iterations=1)

    # find contours
    contours, hierarchy = cv2.findContours(
        dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # Filter overlapping contours
    overlapping = []
    for i, cnt in enumerate(contours):
        for j, cnt2 in enumerate(contours):
            if i != j:
                area1 = cv2.contourArea(cnt)
                area2 = cv2.contourArea(cnt2)
                if area1 > area2:
                    continue
                intersect_area = intersection_area(cnt, cnt2, img)
                if intersect_area > 0.8 * area1:
                    #if area1 > 0.7 * area2 and area1 < 1.3 * area2 :
                    overlapping.append(i)
                    break
    contours = [cnt for i, cnt in enumerate(
        contours) if i not in overlapping]

    # std deviation filter
    # mean_area = np.mean([cv2.contourArea(cnt) for cnt in contours])
    # std_area = np.std([cv2.contourArea(cnt) for cnt in contours])
    # tresh_area = mean_area + std_area

    # final_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > tresh_area]

    # filter in contours that dont have parents
    # final_contours2 = [cnt for i,cnt in enumerate(contours) if hierarchy[0][i][3] == -1]

    # for i,cnt in enumerate(contours):
    #     if hierarchy[0][i][2] == -1:
    #         print(cv2.contourArea(cnt))
    #         final_contours.append(cnt)

    # show image with contours
    # get bounding rect

    return contours

def get_colors(img, contours):
    average_colors = []
    for contour in contours:
        mask = np.zeros(img.shape[:2], np.uint8)
        cv2.drawContours(mask, [contour], -1, (255), -1)
        threshold_distance = 0.07
        mean_color = cv2.mean(img, mask=mask)[:3]  
        if not is_color_similar(mean_color, average_colors, threshold_distance):
            average_colors.append(mean_color)
        mask.fill(0)

    return len(average_colors)

def is_color_similar(color, color_list, threshold):
    """Check if the given color is similar to any color in the list."""
    max_distance = np.sqrt(3 * 255**2)
    for existing_color in color_list:
        color1_np = np.array(color)
        color2_np = np.array(existing_color)
        color_distance= np.sqrt(np.sum((color1_np - color2_np)**2))
        similarity = (max_distance - color_distance) / max_distance
        if similarity > 1 - threshold:
            return True
    return False


if __name__ == '__main__':
    main()
