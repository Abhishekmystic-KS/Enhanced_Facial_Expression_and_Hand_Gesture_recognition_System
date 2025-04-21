import cv2
import numpy as np
import argparse
import time
import os
from utils.precise_hand_detector import PreciseHandDetector

def parse_args():
    """
    Parse command line arguments.

    Returns:
        args: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description='Hand Detection Application')

    parser.add_argument('--camera', type=int, default=0,
                        help='Camera device index')

    parser.add_argument('--width', type=int, default=640,
                        help='Camera frame width')

    parser.add_argument('--height', type=int, default=480,
                        help='Camera frame height')

    parser.add_argument('--min_detection_confidence', type=float, default=0.7,
                        help='Minimum confidence value for hand detection')

    parser.add_argument('--min_tracking_confidence', type=float, default=0.5,
                        help='Minimum confidence value for hand tracking')

    return parser.parse_args()

def main():
    """
    Main function for hand detection application.
    """
    # Parse arguments
    args = parse_args()

    # Initialize hand detector
    detector = PreciseHandDetector(
        min_detection_confidence=args.min_detection_confidence,
        min_tracking_confidence=args.min_tracking_confidence
    )

    # Initialize webcam
    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    print("Hand Detection Application")
    print("Press 'q' to quit")

    while True:
        # Read frame from webcam
        ret, frame = cap.read()

        if not ret:
            print("Error: Failed to capture frame")
            break

        # Flip the frame horizontally for a more natural interaction
        frame = cv2.flip(frame, 1)

        # Find hands
        frame, results = detector.find_hands(frame)

        # Calculate and display FPS
        frame, fps = detector.calculate_fps(frame)

        # Find positions of hand contours if available
        if results and 'contours' in results and len(results['contours']) > 0:
            position_list = detector.find_positions(frame)

            # You can use position_list for additional processing if needed

        # Display the result
        cv2.imshow('Hand Detection', frame)

        # Check for key press
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
