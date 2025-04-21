import os
import cv2
import numpy as np
from tests.direct_test import WeaponDetector, read_image

def test_weapon_detection():
    print("[INFO] Initializing WeaponDetector...")
    detector = WeaponDetector()

    test_image_path = os.path.join(os.path.dirname(__file__), 'test_image.jpg')
    assert os.path.exists(test_image_path), f"Test image not found at {test_image_path}"

    print(f"[INFO] Reading image from {test_image_path}")
    image_np = read_image(test_image_path)

    print("[INFO] Running detection...")
    results = detector.detect(image_np)

    # Assertions
    assert 'boxes' in results and isinstance(results['boxes'], list), "'boxes' key missing or invalid"
    assert 'scores' in results and isinstance(results['scores'], list), "'scores' key missing or invalid"
    assert 'classes' in results and isinstance(results['classes'], list), "'classes' key missing or invalid"
    assert results['annotated_image'] is not None, "Annotated image not generated"
    assert isinstance(results['annotated_image'], bytes), "Annotated image is not in bytes format"

    print(f"[RESULT] Detected {len(results['boxes'])} object(s)")
    print(f"[DETAILS] Scores: {results['scores']}")
    print(f"[DETAILS] Classes: {results['classes']}")

    if results['scores']:
        assert all(score >= 0.5 for score in results['scores']), "Some scores below 0.5 confidence"

    # Save output image
    output_path = os.path.join(os.path.dirname(__file__), 'output.jpg')
    with open(output_path, 'wb') as f:
        f.write(results['annotated_image'])
    print(f"[INFO] Saved annotated image to {output_path}")

if __name__ == '__main__':
    test_weapon_detection()