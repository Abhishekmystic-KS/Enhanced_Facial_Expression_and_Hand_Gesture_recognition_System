import cv2
import numpy as np
import argparse
import time
import os
from utils.improved_hand_utils import ImprovedHandDetector
from utils.face_utils import FaceDetector
from models.expression_model import ExpressionRecognitionModel
from models.gesture_model import GestureRecognitionModel

def parse_args():
    """
    Parse command line arguments.
    
    Returns:
        args: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description='Advanced Hand Gesture Recognition')
    
    parser.add_argument('--expression_model', type=str, default=None,
                        help='Path to pre-trained expression recognition model')
    
    parser.add_argument('--gesture_model', type=str, default=None,
                        help='Path to pre-trained gesture recognition model')
    
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
    
    parser.add_argument('--record', action='store_true',
                        help='Record detected gestures')
    
    parser.add_argument('--output_dir', type=str, default='data/gestures',
                        help='Directory to save recorded gestures')
    
    return parser.parse_args()

def create_advanced_ui(frame, hand_data, ml_gesture=None, ml_confidence=None, 
                      face_expression=None, face_confidence=None, fps=0, recording=False):
    """
    Create an advanced user interface with hand and face information.
    
    Args:
        frame: Input frame.
        hand_data: Hand detection data.
        ml_gesture: Gesture predicted by machine learning model.
        ml_confidence: Confidence of machine learning prediction.
        face_expression: Detected facial expression.
        face_confidence: Confidence of facial expression detection.
        fps: Frames per second.
        recording: Whether recording is active.
        
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
    cv2.rectangle(overlay, (0, h-150), (w, h), (0, 0, 0), -1)  # Bottom panel
    ui_frame = cv2.addWeighted(overlay, 0.7, ui_frame, 0.3, 0)
    
    # Add title
    cv2.putText(ui_frame, "Advanced Hand Gesture Recognition", (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # Add FPS counter
    cv2.putText(ui_frame, f"FPS: {fps:.1f}", (w-150, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Add recording indicator
    if recording:
        cv2.circle(ui_frame, (w-180, 30), 10, (0, 0, 255), -1)
        cv2.putText(ui_frame, "REC", (w-230, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # Add hand information
    if hand_data["hand_found"]:
        # Vision-based detection
        cv2.putText(ui_frame, f"Vision Detection:", (10, h-120), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(ui_frame, f"  Fingers: {hand_data['finger_count']}", (10, h-90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(ui_frame, f"  Gesture: {hand_data['gesture']}", (10, h-60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # ML-based detection
        cv2.putText(ui_frame, f"ML Detection:", (w//2, h-120), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        if ml_gesture is not None:
            cv2.putText(ui_frame, f"  Gesture: {ml_gesture}", (w//2, h-90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Draw confidence bar
            bar_x, bar_y = w//2, h-60
            bar_width, bar_height = 150, 15
            cv2.rectangle(ui_frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (50, 50, 50), -1)
            cv2.rectangle(ui_frame, (bar_x, bar_y), (bar_x + int(bar_width * ml_confidence), bar_y + bar_height), 
                         (0, 255, 0), -1)
            cv2.putText(ui_frame, f"{int(ml_confidence*100)}%", (bar_x + bar_width + 5, bar_y + 12), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        else:
            cv2.putText(ui_frame, "  Waiting for prediction...", (w//2, h-90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    else:
        cv2.putText(ui_frame, "No hand detected", (10, h-90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # Add face information
    if face_expression is not None:
        cv2.putText(ui_frame, f"Expression: {face_expression} ({int(face_confidence*100)}%)", 
                   (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Add instructions
    cv2.putText(ui_frame, "Press 'q' to quit, 'r' to reset, 's' to start/stop recording", (10, h-30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return ui_frame

def main():
    """
    Main function for advanced hand gesture recognition.
    """
    # Parse arguments
    args = parse_args()
    
    # Initialize detectors
    hand_detector = ImprovedHandDetector(threshold=args.threshold, blur_value=args.blur)
    face_detector = FaceDetector(detection_method=args.face_detection)
    
    # Initialize models
    expression_model = ExpressionRecognitionModel(model_path=args.expression_model)
    gesture_model = GestureRecognitionModel(model_path=args.gesture_model)
    
    # Initialize webcam
    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    
    # Initialize variables
    fps = 0
    frame_count = 0
    start_time = time.time()
    recording = False
    current_gesture = None
    gesture_count = 0
    
    # Create output directory if recording is enabled
    if args.record:
        os.makedirs(args.output_dir, exist_ok=True)
    
    print("Advanced Hand Gesture Recognition")
    print("Place your hand in the green rectangle")
    print("Press 'q' to quit, 'r' to reset background, 's' to start/stop recording")
    
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
        
        # Get ML prediction if hand is found
        ml_gesture = None
        ml_confidence = None
        
        if hand_data["hand_found"]:
            # Extract features for ML prediction
            hand_features = hand_detector.get_hand_features(hand_data)
            
            if hand_features is not None:
                # Predict gesture using ML model
                ml_gesture, ml_confidence = gesture_model.predict(hand_features)
                
                # Record gesture if recording is enabled
                if recording and args.record:
                    # Save only if gesture is stable for a few frames
                    if ml_gesture == current_gesture:
                        gesture_count += 1
                        
                        if gesture_count >= 10:  # Save every 10 frames with the same gesture
                            # Create directory for this gesture if it doesn't exist
                            gesture_dir = os.path.join(args.output_dir, ml_gesture.lower().replace(' ', '_'))
                            os.makedirs(gesture_dir, exist_ok=True)
                            
                            # Save hand features
                            feature_file = os.path.join(gesture_dir, f"{ml_gesture.lower().replace(' ', '_')}_{time.time()}.npy")
                            np.save(feature_file, hand_features)
                            
                            # Reset counter
                            gesture_count = 0
                            print(f"Saved {ml_gesture} gesture")
                    else:
                        current_gesture = ml_gesture
                        gesture_count = 0
        
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
        result_frame = create_advanced_ui(
            frame_with_face,
            hand_data,
            ml_gesture=ml_gesture,
            ml_confidence=ml_confidence,
            face_expression=face_expression,
            face_confidence=face_confidence,
            fps=fps,
            recording=recording
        )
        
        # Display the result
        cv2.imshow('Advanced Hand Gesture Recognition', result_frame)
        
        # Check for key press
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            # Reset background
            hand_detector.reset_background()
            print("Background reset")
        elif key == ord('s'):
            # Toggle recording
            recording = not recording
            print(f"Recording {'started' if recording else 'stopped'}")
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
