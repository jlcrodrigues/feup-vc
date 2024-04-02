import json
import os
import unittest
from main import load_input, run_pipeline, ImageResult, DetectedObject

class TestPipelineAccuracy(unittest.TestCase):
    INPUT_DIR = 'test/input/'
    EXPECTED_DIR = 'test/expected/'

    def setUp(self):
        self.input_files = os.listdir(self.INPUT_DIR)

    def test_accuracy(self):
        for input_file in self.input_files:
            if input_file.endswith('.json'):
                input_path = os.path.join(self.INPUT_DIR, input_file)
                expected_path = os.path.join(self.EXPECTED_DIR, input_file)

                # Load input data and run pipeline
                image_files = load_input(input_path)
                results = run_pipeline(image_files)

                # Load expected results
                with open(expected_path) as f:
                    expected_data = json.load(f)

                # Initialize counts for accuracy calculation
                total_detections = 0
                correct_detections = 0
                detection_errors = []
                total_colors = 0
                correct_colors = 0

                # Compare actual results with expected results
                for i, expected_image_result in enumerate(expected_data['results']):
                    actual_image_result = results.results[i]

                    total_colors += expected_image_result['num_colors']
                    correct_colors += actual_image_result.num_colors

                    total_detections += expected_image_result['num_detections']
                    correct_detections += actual_image_result.num_detections

                    detection_errors.append(abs(expected_image_result['num_detections'] - actual_image_result.num_detections)/expected_image_result['num_detections'])

                    # TODO: Compare detected objects
                    

                # Calculate percentages
                accuracy_colors = (correct_colors / total_colors) * 100
                accuracy_detections = (correct_detections / total_detections) * 100
                accuracy_detections = (1 - sum(detection_errors) / len(detection_errors)) * 100

                # Print percentages
                print(f"Accuracy for \033[1m\033[4m{input_file}\033[0m:")
                print(f"  Detections:  \033[1m{accuracy_detections:.2f}%\033[0m ({correct_detections}/{total_detections})")
                print(f"  Colors:      \033[1m{accuracy_colors:.2f}%\033[0m ({correct_colors}/{total_colors})")
                print()

if __name__ == '__main__':
    unittest.main()
