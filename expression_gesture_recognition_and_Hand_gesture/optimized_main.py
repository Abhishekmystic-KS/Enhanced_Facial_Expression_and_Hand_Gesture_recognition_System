import cv2
import numpy as np
import os
import argparse
import time
from utils.face_utils import FaceDetector
from utils.hand_utils import HandDetector
from utils.visualization import create_prediction_overlay
from models.expression_model import ExpressionRecognitionModel
from models.gesture_model import GestureRecognitionModel
from utils.advanced_features import extract_advanced_face_features, extract_advanced_hand_features
from utils.optimization import FrameProcessor, resize_frame, optimize_detection_parameters, MotionDetector

def parse_args():
    """
    Parse command line arguments.
    
    Returns:
        args: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description='Optimized Facial Expression and Hand Gesture Recognition')
    
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
    
    parser.add_argument('--use_advanced_features', action='store_true',
                        help='Use advanced feature extraction')
    
    parser.add_argument('--use_threading', action='store_true',
                        help='Use threading for frame processing')
    
    parser.add_argument('--use_motion_detection', action='store_true',
                        help='Use motion detection to optimize processing')
    
    parser.add_argument('--downscale_factor', type=int, default=100,
                        help='Downscale factor for frame processing (percentage)')
    
    parser.add_argument('--ui_mode', type=str, default='default', choices=['default', 'debug', 'minimal'],
                        help='UI mode')
    
    return parser.parse_args()

def process_frame(frame, face_detector, hand_detector, expression_model, gesture_model, use_advanced_features=False):
    """
    Process a frame for facial expression and hand gesture recognition.
    
    Args:
        frame: Input frame.
        face_detector: Face detector instance.
        hand_detector: Hand detector instance.
        expression_model: Expression recognition model.
        gesture_model: Gesture recognition model.
        use_advanced_features: Whether to use advanced feature extraction.
        
    Returns:
        result: Dictionary with processing results.
    """
    # Create a copy of the frame for processing
    frame_copy = frame.copy()
    
    # Detect faces
    frame_with_faces, face_rects = face_detector.detect_faces(frame_copy)
    
    # Extract face ROI and predict expression
    face_expression = None
    face_confidence = None
    
    if face_rects and len(face_rects) > 0:
        face_roi = face_detector.extract_face_roi(frame, face_rects)
        if face_roi is not None:
            if use_advanced_features:
                # Use advanced features
                face_features = extract_advanced_face_features(face_roi)
                # For now, we'll still use the basic model
                face_expression, face_confidence = expression_model.predict(face_roi)
            else:
                face_expression, face_confidence = expression_model.predict(face_roi)
    
    # Detect hands
    frame_with_hands, hand_rects = hand_detector.detect_hands(frame_with_faces)
    
    # Extract hand features and predict gesture
    hand_gesture = None
    hand_confidence = None
    
    if hand_rects and len(hand_rects) > 0:
        if use_advanced_features:
            # Use advanced features
            hand_features = extract_advanced_hand_features(hand_rects, frame)
            if hand_features and len(hand_features) > 0:
                # For now, we'll still use the basic model with basic features
                basic_features = hand_detector.get_hand_gesture_features(hand_rects)
                if basic_features and len(basic_features) > 0:
                    hand_gesture, hand_confidence = gesture_model.predict(basic_features[0])
        else:
            hand_features = hand_detector.get_hand_gesture_features(hand_rects)
            if hand_features and len(hand_features) > 0:
                hand_gesture, hand_confidence = gesture_model.predict(hand_features[0])
    
    # Return results
    return {
        'frame': frame_with_hands,
        'face_expression': face_expression,
        'face_confidence': face_confidence,
        'hand_gesture': hand_gesture,
        'hand_confidence': hand_confidence
    }

def create_enhanced_ui(frame, face_expression=None, face_confidence=None, hand_gesture=None, hand_confidence=None, 
                      fps=0, mode='default', stats=None):
    """
    Create an enhanced UI with more visualization options.
    
    Args:
        frame: The input frame.
        face_expression: Predicted facial expression.
        face_confidence: Confidence of facial expression prediction.
        hand_gesture: Predicted hand gesture.
        hand_confidence: Confidence of hand gesture prediction.
        fps: Frames per second.
        mode: Display mode ('default', 'debug', 'minimal').
        stats: Additional statistics to display.
        
    Returns:
        ui_frame: Frame with enhanced UI.
    """
    # Create a copy of the frame
    ui_frame = frame.copy()
    
    # Get frame dimensions
    h, w = ui_frame.shape[:2]
    
    # Create a semi-transparent overlay for the UI panel
    overlay = ui_frame.copy()
    
    if mode == 'minimal':
        # Minimal mode - just show predictions in a small corner
        if face_expression is not None:
            cv2.putText(ui_frame, f"Face: {face_expression} ({int(face_confidence*100)}%)", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        if hand_gesture is not None:
            cv2.putText(ui_frame, f"Hand: {hand_gesture} ({int(hand_confidence*100)}%)", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    elif mode == 'debug':
        # Debug mode - show detailed information
        # Draw panel background
        cv2.rectangle(overlay, (0, 0), (w, 120), (0, 0, 0), -1)
        cv2.rectangle(overlay, (w-200, 0), (w, h), (0, 0, 0), -1)
        
        # Add transparency
        ui_frame = cv2.addWeighted(overlay, 0.7, ui_frame, 0.3, 0)
        
        # Add FPS counter
        cv2.putText(ui_frame, f"FPS: {fps:.1f}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Add face expression info
        if face_expression is not None:
            cv2.putText(ui_frame, f"Expression: {face_expression}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Draw confidence bar
            bar_x, bar_y = 10, 70
            bar_width, bar_height = 150, 15
            cv2.rectangle(ui_frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (50, 50, 50), -1)
            cv2.rectangle(ui_frame, (bar_x, bar_y), (bar_x + int(bar_width * face_confidence), bar_y + bar_height), 
                         (0, 255, 0), -1)
            cv2.putText(ui_frame, f"{int(face_confidence*100)}%", (bar_x + bar_width + 5, bar_y + 12), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Add hand gesture info
        if hand_gesture is not None:
            cv2.putText(ui_frame, f"Gesture: {hand_gesture}", (10, 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Draw confidence bar
            bar_x, bar_y = 10, 110
            bar_width, bar_height = 150, 15
            cv2.rectangle(ui_frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (50, 50, 50), -1)
            cv2.rectangle(ui_frame, (bar_x, bar_y), (bar_x + int(bar_width * hand_confidence), bar_y + bar_height), 
                         (0, 255, 0), -1)
            cv2.putText(ui_frame, f"{int(hand_confidence*100)}%", (bar_x + bar_width + 5, bar_y + 12), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Add system info
        cv2.putText(ui_frame, "System Info:", (w-190, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(ui_frame, f"Frame: {w}x{h}", (w-190, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Add stats if available
        if stats:
            y_pos = 90
            for key, value in stats.items():
                if isinstance(value, float):
                    text = f"{key}: {value:.2f}"
                else:
                    text = f"{key}: {value}"
                cv2.putText(ui_frame, text, (w-190, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                y_pos += 30
        
        # Add controls info
        cv2.putText(ui_frame, "Controls:", (w-190, h-120), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(ui_frame, "q - Quit", (w-190, h-90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(ui_frame, "r - Reset background", (w-190, h-60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(ui_frame, "m - Change mode", (w-190, h-30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    else:  # default mode
        # Draw panel background
        cv2.rectangle(overlay, (0, h-100), (w, h), (0, 0, 0), -1)
        
        # Add transparency
        ui_frame = cv2.addWeighted(overlay, 0.7, ui_frame, 0.3, 0)
        
        # Add face expression info
        if face_expression is not None:
            cv2.putText(ui_frame, f"Expression: {face_expression} ({int(face_confidence*100)}%)", 
                       (10, h-70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Add hand gesture info
        if hand_gesture is not None:
            cv2.putText(ui_frame, f"Gesture: {hand_gesture} ({int(hand_confidence*100)}%)", 
                       (10, h-30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Add FPS counter
        cv2.putText(ui_frame, f"FPS: {fps:.1f}", (w-150, h-70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Add controls info
        cv2.putText(ui_frame, "q - Quit | r - Reset | m - Mode", (w-350, h-30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return ui_frame

def main():
    """
    Main function for the optimized application.
    """
    # Parse arguments
    args = parse_args()
    
    # Initialize detectors
    face_detector = FaceDetector(detection_method=args.face_detection)
    hand_detector = HandDetector()
    
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
    mode = args.ui_mode
    
    # Initialize motion detector if enabled
    motion_detector = None
    if args.use_motion_detection:
        motion_detector = MotionDetector()
    
    # Initialize frame processor if threading is enabled
    frame_processor = None
    if args.use_threading:
        frame_processor = FrameProcessor()
        frame_processor.start(
            lambda frame: process_frame(
                frame, face_detector, hand_detector, 
                expression_model, gesture_model, 
                args.use_advanced_features
            )
        )
    
    print("Optimized Facial Expression and Hand Gesture Recognition")
    print("Press 'q' to quit, 'r' to reset background, 'm' to change UI mode")
    
    while True:
        # Read frame from webcam
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Failed to capture frame")
            break
        
        # Downscale frame if needed
        if args.downscale_factor < 100:
            frame = resize_frame(frame, args.downscale_factor)
        
        # Process frame
        if args.use_threading:
            # Add frame to processing queue
            frame_processor.add_frame(frame)
            
            # Get processed result
            result = frame_processor.get_result()
            
            if result:
                # Update UI
                stats = frame_processor.get_stats()
                result_frame = create_enhanced_ui(
                    result['frame'],
                    face_expression=result['face_expression'],
                    face_confidence=result['face_confidence'],
                    hand_gesture=result['hand_gesture'],
                    hand_confidence=result['hand_confidence'],
                    fps=stats['fps'],
                    mode=mode,
                    stats=stats
                )
                
                # Display the result
                cv2.imshow('Optimized Expression and Gesture Recognition', result_frame)
        else:
            # Use motion detection if enabled
            if args.use_motion_detection and motion_detector:
                motion_detected, motion_regions = motion_detector.detect_motion(frame)
                
                # Only process frame if motion is detected
                if motion_detected or frame_count % 10 == 0:  # Process every 10th frame even without motion
                    # Process frame
                    result = process_frame(
                        frame, face_detector, hand_detector, 
                        expression_model, gesture_model, 
                        args.use_advanced_features
                    )
                    
                    # Calculate FPS
                    frame_count += 1
                    elapsed_time = time.time() - start_time
                    if elapsed_time > 1.0:
                        fps = frame_count / elapsed_time
                        frame_count = 0
                        start_time = time.time()
                    
                    # Update UI
                    result_frame = create_enhanced_ui(
                        result['frame'],
                        face_expression=result['face_expression'],
                        face_confidence=result['face_confidence'],
                        hand_gesture=result['hand_gesture'],
                        hand_confidence=result['hand_confidence'],
                        fps=fps,
                        mode=mode
                    )
                    
                    # Display the result
                    cv2.imshow('Optimized Expression and Gesture Recognition', result_frame)
                else:
                    # Just display the frame with UI
                    result_frame = create_enhanced_ui(
                        frame,
                        fps=fps,
                        mode=mode
                    )
                    
                    # Display the result
                    cv2.imshow('Optimized Expression and Gesture Recognition', result_frame)
            else:
                # Process frame without motion detection
                result = process_frame(
                    frame, face_detector, hand_detector, 
                    expression_model, gesture_model, 
                    args.use_advanced_features
                )
                
                # Calculate FPS
                frame_count += 1
                elapsed_time = time.time() - start_time
                if elapsed_time > 1.0:
                    fps = frame_count / elapsed_time
                    frame_count = 0
                    start_time = time.time()
                
                # Update UI
                result_frame = create_enhanced_ui(
                    result['frame'],
                    face_expression=result['face_expression'],
                    face_confidence=result['face_confidence'],
                    hand_gesture=result['hand_gesture'],
                    hand_confidence=result['hand_confidence'],
                    fps=fps,
                    mode=mode
                )
                
                # Display the result
                cv2.imshow('Optimized Expression and Gesture Recognition', result_frame)
        
        # Check for key press
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            # Reset background subtractor for hand detection
            hand_detector = HandDetector()
            if motion_detector:
                motion_detector = MotionDetector()
            print("Background reset for hand detection")
        elif key == ord('m'):
            # Change UI mode
            if mode == 'default':
                mode = 'debug'
            elif mode == 'debug':
                mode = 'minimal'
            else:
                mode = 'default'
            print(f"UI mode changed to: {mode}")
    
    # Release resources
    cap.release()
    if args.use_threading and frame_processor:
        frame_processor.stop()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
