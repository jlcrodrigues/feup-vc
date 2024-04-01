import json
from dataclasses import dataclass, asdict
from typing import List
import cv2 
import np

# INPUT_PATH = 'json-example-task1/input.json'
INPUT_PATH = 'test/input/demo.json'
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

def run_pipeline(samples: List[str]) -> Results:
    results = []
    for sample in samples:
        results.append(process_image(sample))
    return Results(results)

def process_image(sample_path) -> ImageResult:
    # TODO: Implement image processing
    num_detections = num_of_pieces(sample_path)

    
    return ImageResult(
        file_name=sample_path,
        num_colors=0,
        num_detections=num_detections,
        detected_objects=[]
    )

def grayscale_conversion(img):
    cv2.imshow('Original Image', img)
    cv2.waitKey(0)
    cv2.destroyWindow('Original Image')

    # Convert image to grayscale
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow('Grayscale Image', gray_image)
    cv2.waitKey(0)
    cv2.destroyWindow('Grayscale Image')

    # return gray_image
    return gray_image



def num_of_pieces(sample_path):
    img = cv2.imread(SAMPLES_PATH + sample_path)

    # resize image preserving aspect ratio
    img = cv2.resize(img, (800, 800))

    # Convert image to grayscale
    gray_img = grayscale_conversion(img)

    # apply blur
    blurred_img = cv2.GaussianBlur(gray_img, (81, 81), 0)
    cv2.imshow('Blurred Image', blurred_img)
    cv2.waitKey(0)
    cv2.destroyWindow('Blurred Image')

    # apply canny
    # canny_img = cv2.Canny(blurred_img, 100, 200)
    # cv2.imshow('Canny Image', canny_img)
    # cv2.waitKey(0)
    # cv2.destroyWindow('Canny Image')
    

    # # Apply threshold
    tresh = cv2.threshold(gray_img, 130, 255, cv2.THRESH_BINARY)[1]
    cv2.imshow('Threshold Image 100', tresh)
    cv2.waitKey(0)
    cv2.destroyWindow('Threshold Image 100')

    # # apply erosions to reduce size of foreground objects
    # mask = tresh.copy()
    # mask = cv2.erode(mask, None, iterations=5)
    # cv2.imshow('Eroded Image', mask)
    # cv2.waitKey(0)
    # cv2.destroyWindow('Eroded Image')

    # apply canny edge filter
    mask = cv2.Canny(tresh, 150, 250)
    cv2.imshow('Canny Image', mask)
    cv2.waitKey(0)
    cv2.destroyWindow('Canny Image')

    dilated = cv2.dilate(mask, None, iterations=1)
    cv2.imshow('Dilated Image', dilated)
    cv2.waitKey(0)
    cv2.destroyWindow('Dilated Image')

    # find contours
    contours, hierarchy = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)



    # filter out contours when area is less than 40% of the largest contour
    percentile = 0.5
    tresh_area = cv2.contourArea(contours[0]) * percentile
    print("Treshold area: " + str(tresh_area))

    # std deviation filter
    # mean_area = np.mean([cv2.contourArea(cnt) for cnt in contours])
    # std_area = np.std([cv2.contourArea(cnt) for cnt in contours])
    # tresh_area = mean_area + std_area

    final_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > tresh_area] 
    for cnt in final_contours:
        print(cv2.contourArea(cnt))
    print(cv2.contourArea(contours[0]))

    # filter in contours that dont have parents
    # final_contours2 = [cnt for i,cnt in enumerate(contours) if hierarchy[0][i][3] == -1]


    # for i,cnt in enumerate(contours):
    #     if hierarchy[0][i][2] == -1:
    #         print(cv2.contourArea(cnt))
    #         final_contours.append(cnt)

    # show image with contours
    cv2.drawContours(img, final_contours, -1, (0, 255, 0), 2)
    cv2.imshow('Contours', img)
    cv2.waitKey(0)

    print("Number of pieces: " + str(len(final_contours)))

    return len(final_contours)





def main():
    sample_paths = load_input(INPUT_PATH)
    result = run_pipeline(sample_paths)
    store_output(OUTPUT_PATH, result)

if __name__ == '__main__':
    main()
