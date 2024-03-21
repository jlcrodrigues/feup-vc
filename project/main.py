import json
from dataclasses import dataclass, asdict
from typing import List

INPUT_PATH = 'json-example-task1/input.json'
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
    with open(output_path, 'x') as f:
        json.dump(asdict(results), f)

def run_pipeline(samples: List[str]) -> Results:
    results = []
    for sample in samples:
        results.append(process_image(sample))
    return Results(results)

def process_image(sample_path) -> ImageResult:
    # TODO: Implement image processing
    return ImageResult(
        file_name=sample_path,
        num_colors=0,
        num_detections=0,
        detected_objects=[]
    )

def main():
    sample_paths = load_input(INPUT_PATH)
    result = run_pipeline(sample_paths)
    store_output(OUTPUT_PATH, result)

if __name__ == '__main__':
    main()
