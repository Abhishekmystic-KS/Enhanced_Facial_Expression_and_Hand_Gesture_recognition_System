import cv2
import argparse
import time
from utils.simple_hand_detector import SimpleHandDetector

def parse_args():
    """
    Parse command line arguments.
    
    Returns:
        args: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description='Simple Hand Detection Application')
    
    parser.add_argument('--camera', type=int, default=0,
                        help='Camera device index')
    
    parser.add_argument('--width', type=int, default=640,
                        help='Camera frame width')
    
    parser.add_argument('--height', type=int, default=480,
                        help='Camera frame height')
    
    parser.add_argument('--threshold', type=int, default=20,
                        help='Threshold for hand detection')
    
    parser.add_argument('--blur', type=int, default=7,
                        help='Blur value for hand detection')
    
    parser.add_argument('--min_area', type=int, default=5000,
                        help='Minimum contour area to be considered a hand')
    
    parser.add_argument('--max_area', type=int, default=50000,
                        help='Maximum contour area to be considered a hand')
    
    return parser.parse_args()

def main():
    """
    Main function for simple hand detection application.
    """
    # Parse arguments
    args = parse_args()
    
    # Initialize hand detector
    detector = SimpleHandDetector(
        threshold=args.threshold,
        blur_value=args.blur,
        min_area=args.min_area,
        max_area=args.max_area
    )
    
    # Initialize webcam
    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    
    print("Simple Hand Detection Application")
    print("Press 'r' to reset background")
    print("Press 'q' to quit")
    
    # Wait a moment to initialize
    time.sleep(1)
    
    # Reset background
    detector.reset_background()
    
    while True:
        # Read frame from webcam
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Failed to capture frame")
            break
        
        # Flip the frame horizontally for a more natural interaction
        frame = cv2.flip(frame, 1)
        
        # Detect hand
        frame, hand_data = detector.detect_hand(frame)
        
        # Calculate and display FPS
        frame, fps = detector.calculate_fps(frame)
        
        # Display the result
        cv2.imshow('Simple Hand Detection', frame)
        
        # Check for key press
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            print("Resetting background...")
            detector.reset_background()
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
