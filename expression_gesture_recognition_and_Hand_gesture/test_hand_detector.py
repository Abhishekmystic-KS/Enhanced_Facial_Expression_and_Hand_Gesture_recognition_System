import cv2
import argparse
from utils.precise_hand_detector import PreciseHandDetector

def parse_args():
    """
    Parse command line arguments.

    Returns:
        args: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description='Test Hand Detector on Image')

    parser.add_argument('--image', type=str, required=True,
                        help='Path to the input image')

    parser.add_argument('--output', type=str, default='output.jpg',
                        help='Path to save the output image')

    parser.add_argument('--min_detection_confidence', type=float, default=0.7,
                        help='Minimum confidence value for hand detection')

    return parser.parse_args()

def main():
    """
    Main function for testing hand detector on an image.
    """
    # Parse arguments
    args = parse_args()

    # Initialize hand detector
    detector = PreciseHandDetector(
        static_image_mode=True,
        min_detection_confidence=args.min_detection_confidence
    )

    # Read input image
    image = cv2.imread(args.image)

    if image is None:
        print(f"Error: Could not read image from {args.image}")
        return

    # Find hands
    image, results = detector.find_hands(image)

    # Calculate and display FPS (just for consistency with the app)
    image, _ = detector.calculate_fps(image)

    # Save output image
    cv2.imwrite(args.output, image)
    print(f"Output saved to {args.output}")

    # Display the result
    cv2.imshow('Hand Detection', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
