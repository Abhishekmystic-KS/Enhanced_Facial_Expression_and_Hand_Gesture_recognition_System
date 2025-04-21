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

def parse_args():
    """
    Parse command line arguments.
    
    Returns:
        args: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description='Enhanced UI for Facial Expression and Hand Gesture Recognition')
    
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
    
    parser.add_argument('--record', action='store_true',
                        help='Record the session')
    
    return parser.parse_args()

def create_enhanced_ui(frame, face_expression=None, face_confidence=None, hand_gesture=None, hand_confidence=None, 
                      fps=0, mode='default', history=None):
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
        history: History of predictions for trend display.
        
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
    
    # Add history visualization if available
    if history is not None and len(history) > 0:
        # Create a small graph of recent predictions
        graph_width, graph_height = 200, 100
        graph = np.ones((graph_height, graph_width, 3), dtype=np.uint8) * 255
        
        # Draw graph background
        cv2.rectangle(graph, (0, 0), (graph_width, graph_height), (240, 240, 240), -1)
        
        # Draw grid lines
        for i in range(0, graph_width, 20):
            cv2.line(graph, (i, 0), (i, graph_height), (220, 220, 220), 1)
        for i in range(0, graph_height, 20):
            cv2.line(graph, (0, i), (graph_width, i), (220, 220, 220), 1)
        
        # Draw confidence history
        if len(history) > 1:
            for i in range(1, min(len(history), graph_width)):
                if history[i-1] is not None and history[i] is not None:
                    pt1 = (graph_width - i, graph_height - int(history[i-1] * graph_height))
                    pt2 = (graph_width - i + 1, graph_height - int(history[i] * graph_height))
                    cv2.line(graph, pt1, pt2, (0, 0, 255), 2)
        
        # Place the graph in the corner
        ui_frame[10:10+graph_height, w-graph_width-10:w-10] = graph
    
    return ui_frame

def main():
    """
    Main function for the enhanced UI application.
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
    mode = 'default'  # UI mode: 'default', 'debug', 'minimal'
    
    # Initialize history for trend display
    face_history = []
    hand_history = []
    
    # Initialize video writer if recording
    video_writer = None
    if args.record:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        output_path = f"recording_{time.strftime('%Y%m%d_%H%M%S')}.avi"
        video_writer = cv2.VideoWriter(output_path, fourcc, 20.0, (args.width, args.height))
        print(f"Recording to {output_path}")
    
    print("Enhanced UI for Facial Expression and Hand Gesture Recognition")
    print("Press 'q' to quit, 'r' to reset background, 'm' to change UI mode")
    
    while True:
        # Read frame from webcam
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Failed to capture frame")
            break
        
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
                if args.use_advanced_features:
                    # Use advanced features
                    face_features = extract_advanced_face_features(face_roi)
                    # For now, we'll still use the basic model
                    face_expression, face_confidence = expression_model.predict(face_roi)
                else:
                    face_expression, face_confidence = expression_model.predict(face_roi)
                
                # Update face history
                face_history.append(face_confidence)
                if len(face_history) > 100:
                    face_history.pop(0)
        
        # Detect hands
        frame_with_hands, hand_rects = hand_detector.detect_hands(frame_with_faces)
        
        # Extract hand features and predict gesture
        hand_gesture = None
        hand_confidence = None
        
        if hand_rects and len(hand_rects) > 0:
            if args.use_advanced_features:
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
            
            # Update hand history
            if hand_confidence is not None:
                hand_history.append(hand_confidence)
                if len(hand_history) > 100:
                    hand_history.pop(0)
        
        # Calculate FPS
        frame_count += 1
        elapsed_time = time.time() - start_time
        if elapsed_time > 1.0:
            fps = frame_count / elapsed_time
            frame_count = 0
            start_time = time.time()
        
        # Create enhanced UI
        result_frame = create_enhanced_ui(
            frame_with_hands,
            face_expression=face_expression,
            face_confidence=face_confidence,
            hand_gesture=hand_gesture,
            hand_confidence=hand_confidence,
            fps=fps,
            mode=mode,
            history=face_history
        )
        
        # Record if enabled
        if args.record and video_writer is not None:
            video_writer.write(result_frame)
        
        # Display the result
        cv2.imshow('Enhanced Expression and Gesture Recognition', result_frame)
        
        # Check for key press
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            # Reset background subtractor for hand detection
            hand_detector = HandDetector()
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
    if video_writer is not None:
        video_writer.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
