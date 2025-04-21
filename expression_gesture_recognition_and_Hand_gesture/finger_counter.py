import cv2
import numpy as np
import argparse
import time
from utils.improved_hand_utils import ImprovedHandDetector
from utils.face_utils import FaceDetector
from models.expression_model import ExpressionRecognitionModel

def parse_args():
    """
    Parse command line arguments.
    
    Returns:
        args: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description='Finger Counter and Hand Gesture Recognition')
    
    parser.add_argument('--expression_model', type=str, default=None,
                        help='Path to pre-trained expression recognition model')
    
    parser.add_argument('--face_detection', type=str, default='haar', choices=['haar', 'dnn'],
                        help='Face detection method (haar or dnn)')
    
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

def create_ui(frame, hand_data, face_expression=None, face_confidence=None, fps=0):
    """
    Create a user interface with hand and face information.
    
    Args:
        frame: Input frame.
        hand_data: Hand detection data.
        face_expression: Detected facial expression.
        face_confidence: Confidence of facial expression detection.
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
    cv2.rectangle(overlay, (0, h-150), (w, h), (0, 0, 0), -1)
    ui_frame = cv2.addWeighted(overlay, 0.7, ui_frame, 0.3, 0)
    
    # Add hand information
    if hand_data["hand_found"]:
        cv2.putText(ui_frame, f"Fingers: {hand_data['finger_count']}", (10, h-120), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(ui_frame, f"Gesture: {hand_data['gesture']}", (10, h-80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    else:
        cv2.putText(ui_frame, "No hand detected", (10, h-120), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # Add face information
    if face_expression is not None:
        cv2.putText(ui_frame, f"Expression: {face_expression} ({int(face_confidence*100)}%)", 
                   (10, h-40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # Add FPS counter
    cv2.putText(ui_frame, f"FPS: {fps:.1f}", (w-150, h-120), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # Add instructions
    cv2.putText(ui_frame, "Press 'q' to quit, 'r' to reset background", (w-450, h-40), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    return ui_frame

def main():
    """
    Main function for finger counting and hand gesture recognition.
    """
    # Parse arguments
    args = parse_args()
    
    # Initialize detectors
    hand_detector = ImprovedHandDetector(threshold=args.threshold, blur_value=args.blur)
    face_detector = FaceDetector(detection_method=args.face_detection)
    
    # Initialize expression model
    expression_model = ExpressionRecognitionModel(model_path=args.expression_model)
    
    # Initialize webcam
    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    
    # Initialize variables
    fps = 0
    frame_count = 0
    start_time = time.time()
    
    print("Finger Counter and Hand Gesture Recognition")
    print("Place your hand in the green rectangle")
    print("Press 'q' to quit, 'r' to reset background")
    
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
        
        # Detect face and expression
        frame_with_face, face_rects = face_detector.detect_faces(frame_with_hand)
        
        face_expression = None
        face_confidence = None
        
        if face_rects and len(face_rects) > 0:
            face_roi = face_detector.extract_face_roi(frame, face_rects)
            if face_roi is not None:
                face_expression, face_confidence = expression_model.predict(face_roi)
        
        # Calculate FPS
        frame_count += 1
        elapsed_time = time.time() - start_time
        if elapsed_time > 1.0:
            fps = frame_count / elapsed_time
            frame_count = 0
            start_time = time.time()
        
        # Create UI
        result_frame = create_ui(
            frame_with_face,
            hand_data,
            face_expression=face_expression,
            face_confidence=face_confidence,
            fps=fps
        )
        
        # Display the result
        cv2.imshow('Finger Counter and Hand Gesture Recognition', result_frame)
        
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
