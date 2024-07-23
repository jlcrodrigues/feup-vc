""" Assuming an input JSON file with the following format:
{
  "image_files": [
    "photos/1/photo_path.jpg"
  ]
}
"""
import json
from dataclasses import dataclass, asdict
from typing import List
import cv2
import numpy as np
import sys
import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import densenet121, DenseNet121_Weights
from PIL import Image

OUTPUT_PATH = 'output.json'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@dataclass
class ImageResult:
    file_name: str
    num_detections: int


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

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow('Image with contours', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    """
    Main function to run the program.
    The program expects two arguments:
    - input_path: Path to the input JSON file
    """
    if len(sys.argv) < 2:
        print("Usage: python main.py <input_file>")
        print()
        print("Example:")
        print("  python main.py input.json")
        sys.exit(1)
    
    input_path = sys.argv[1]

    print("Loading model architecture...")
    # Load model and set to evaluation mode
    model = densenet121(weights=DenseNet121_Weights.DEFAULT)
    num_features = model.classifier.in_features
    model.classifier = nn.Linear(num_features, 1)
    
    model = model.to(device)
    model.eval()
    
    print("Loading model weights...")
    try:
        model.load_state_dict(torch.load('model.pth', map_location=device))
    except Exception as e:
        print(f"Error loading the model: {e}")
        sys.exit(1)

    sample_paths = load_input(input_path)

    print("Processing images...")
    results = []
    for sample_path in sample_paths:
        image_filename = sample_path

        # Read image
        image = cv2.imread(image_filename)

        # Convert from BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((224, 224)),
                transforms.Normalize([0.485, 0.456, 0.406], [
                                     0.229, 0.224, 0.225])
            ]
        )
        image_tensor = transform(image).to(device)

        with torch.no_grad():
            output = model(image_tensor.unsqueeze(0))
        grounded_predictions = [min(32, max(1, int(round(x.item())))) for x in output]
        num_detections = grounded_predictions[0]

        result = ImageResult(file_name=image_filename, num_detections=num_detections)
        results.append(result)
    
    final_results = Results(results=results)
    print("Storing results to output.json")
    store_output(OUTPUT_PATH, final_results)


if __name__ == '__main__':
    main()
