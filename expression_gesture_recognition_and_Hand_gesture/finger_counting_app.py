import cv2
import numpy as np
import argparse
import time
from utils.improved_hand_utils import ImprovedHandDetector

def parse_args():
    """
    Parse command line arguments.
    
    Returns:
        args: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description='Finger Counting Application')
    
    parser.add_argument('--camera', type=int, default=0,
                        help='Camera device index')
    
    parser.add_argument('--width', type=int, default=640,
                        help='Camera frame width')
    
    parser.add_argument('--height', type=int, default=480,
                        help='Camera frame height')
    
    parser.add_argument('--threshold', type=int, default=60,
                        help='Threshold for hand detection')
    
    parser.add_argument('--blur', type=int, default=41,
                        help='Blur value for hand detection')
    
    return parser.parse_args()

def create_finger_counting_ui(frame, hand_data, fps=0):
    """
    Create a user interface focused on finger counting.
    
    Args:
        frame: Input frame.
        hand_data: Hand detection data.
        fps: Frames per second.
        
    Returns:
        ui_frame: Frame with UI elements.
    """
    # Create a copy of the frame
    ui_frame = frame.copy()
    
    # Get frame dimensions
    h, w = ui_frame.shape[:2]
    
    # Create a semi-transparent overlay for the UI panel
    overlay = ui_frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 100), (0, 0, 0), -1)  # Top panel
    ui_frame = cv2.addWeighted(overlay, 0.7, ui_frame, 0.3, 0)
    
    # Add title
    cv2.putText(ui_frame, "Finger Counting Application", (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # Add FPS counter
    cv2.putText(ui_frame, f"FPS: {fps:.1f}", (w-150, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Add instructions
    cv2.putText(ui_frame, "Place your hand in the green rectangle", (10, 70), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # If hand is detected, show finger count in a large format
    if hand_data["hand_found"]:
        # Draw a large circle in the center with finger count
        circle_center = (w // 2, h // 2)
        circle_radius = min(w, h) // 4
        
        # Draw circle background
        cv2.circle(ui_frame, circle_center, circle_radius, (0, 0, 0), -1)
        cv2.circle(ui_frame, circle_center, circle_radius, (0, 255, 0), 5)
        
        # Draw finger count
        finger_count = hand_data["finger_count"]
        cv2.putText(ui_frame, str(finger_count), 
                   (circle_center[0] - 40, circle_center[1] + 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 5, (255, 255, 255), 10)
        
        # Draw gesture name
        cv2.putText(ui_frame, hand_data["gesture"], 
                   (circle_center[0] - len(hand_data["gesture"]) * 10, circle_center[1] + circle_radius + 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Draw finger landmarks
        for i, landmark in enumerate(hand_data["landmarks"]):
            cv2.circle(ui_frame, landmark, 8, (0, 0, 255), -1)
            cv2.putText(ui_frame, str(i), landmark, 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    else:
        # Show "No hand detected" message
        cv2.putText(ui_frame, "No hand detected", (w // 2 - 150, h // 2), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)
    
    # Add controls info at the bottom
    cv2.putText(ui_frame, "Press 'q' to quit, 'r' to reset background", (10, h - 20), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    return ui_frame

def main():
    """
    Main function for finger counting application.
    """
    # Parse arguments
    args = parse_args()
    
    # Initialize hand detector
    hand_detector = ImprovedHandDetector(threshold=args.threshold, blur_value=args.blur)
    
    # Initialize webcam
    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    
    # Initialize variables
    fps = 0
    frame_count = 0
    start_time = time.time()
    
    print("Finger Counting Application")
    print("Place your hand in the green rectangle")
    print("Press 'q' to quit, 'r' to reset background")
    
    # Wait a moment to initialize
    time.sleep(1)
    
    while True:
        # Read frame from webcam
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Failed to capture frame")
            break
        
        # Flip the frame horizontally for a more natural interaction
        frame = cv2.flip(frame, 1)
        
        # Detect hand
        frame_with_hand, hand_data = hand_detector.detect_hand(frame)
        
        # Calculate FPS
        frame_count += 1
        elapsed_time = time.time() - start_time
        if elapsed_time > 1.0:
            fps = frame_count / elapsed_time
            frame_count = 0
            start_time = time.time()
        
        # Create UI
        result_frame = create_finger_counting_ui(
            frame_with_hand,
            hand_data,
            fps=fps
        )
        
        # Display the result
        cv2.imshow('Finger Counting Application', result_frame)
        
        # Check for key press
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            # Reset background
            hand_detector.reset_background()
            print("Background reset")
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
