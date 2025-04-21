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
    parser = argparse.ArgumentParser(description='Capture a single frame with simple hand detection')
    
    parser.add_argument('--camera', type=int, default=0,
                        help='Camera device index')
    
    parser.add_argument('--width', type=int, default=640,
                        help='Camera frame width')
    
    parser.add_argument('--height', type=int, default=480,
                        help='Camera frame height')
    
    parser.add_argument('--output', type=str, default='simple_hand_detection_output.jpg',
                        help='Output image file path')
    
    parser.add_argument('--delay', type=int, default=3,
                        help='Delay in seconds before capturing the frame')
    
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
    Main function for capturing a single frame with simple hand detection.
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
    
    print(f"Capturing frame in {args.delay} seconds...")
    print("Please position your hand in front of the camera")
    
    # Initialize background
    print("Initializing background...")
    ret, frame = cap.read()
    if ret:
        frame = cv2.flip(frame, 1)
        detector.reset_background()
        # Process a few frames to initialize the background model
        for _ in range(10):
            ret, frame = cap.read()
            if ret:
                frame = cv2.flip(frame, 1)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                blur = cv2.GaussianBlur(gray, (detector.blur_value, detector.blur_value), 0)
                cv2.accumulateWeighted(blur, detector.background, 0.5)
    
    # Wait for the specified delay
    start_time = time.time()
    while time.time() - start_time < args.delay:
        # Read frame from webcam
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Failed to capture frame")
            break
        
        # Flip the frame horizontally for a more natural interaction
        frame = cv2.flip(frame, 1)
        
        # Display countdown
        seconds_left = int(args.delay - (time.time() - start_time))
        cv2.putText(frame, f"Capturing in: {seconds_left}s", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # Display the frame
        cv2.imshow('Capture Hand Frame', frame)
        
        # Check for key press
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    
    # Capture the final frame
    ret, frame = cap.read()
    
    if ret:
        # Flip the frame horizontally
        frame = cv2.flip(frame, 1)
        
        # Process the frame with hand detection
        processed_frame, hand_data = detector.detect_hand(frame)
        
        # Calculate and display FPS
        processed_frame, _ = detector.calculate_fps(processed_frame)
        
        # Save the processed frame
        cv2.imwrite(args.output, processed_frame)
        print(f"Frame saved to {args.output}")
        
        # Display the final frame
        cv2.imshow('Captured Frame', processed_frame)
        cv2.waitKey(0)
    else:
        print("Error: Failed to capture final frame")
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
