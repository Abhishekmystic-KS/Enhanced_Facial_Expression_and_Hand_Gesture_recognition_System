import cv2
import argparse
import time
from utils.precise_hand_detector import PreciseHandDetector

def parse_args():
    """
    Parse command line arguments.
    
    Returns:
        args: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description='Capture a single frame with hand detection')
    
    parser.add_argument('--camera', type=int, default=0,
                        help='Camera device index')
    
    parser.add_argument('--width', type=int, default=640,
                        help='Camera frame width')
    
    parser.add_argument('--height', type=int, default=480,
                        help='Camera frame height')
    
    parser.add_argument('--output', type=str, default='hand_detection_output.jpg',
                        help='Output image file path')
    
    parser.add_argument('--delay', type=int, default=3,
                        help='Delay in seconds before capturing the frame')
    
    parser.add_argument('--min_detection_confidence', type=float, default=0.7,
                        help='Minimum confidence value for hand detection')
    
    return parser.parse_args()

def main():
    """
    Main function for capturing a single frame with hand detection.
    """
    # Parse arguments
    args = parse_args()
    
    # Initialize hand detector
    detector = PreciseHandDetector(
        min_detection_confidence=args.min_detection_confidence
    )
    
    # Initialize webcam
    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    
    print(f"Capturing frame in {args.delay} seconds...")
    print("Please position your hand in front of the camera")
    
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
        processed_frame, results = detector.find_hands(frame)
        
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
