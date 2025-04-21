import cv2
import os
import numpy as np
import argparse
import time
from utils.face_utils import FaceDetector
from utils.hand_utils import HandDetector

def parse_args():
    """
    Parse command line arguments.
    
    Returns:
        args: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description='Data Collection for Expression and Gesture Recognition')
    
    parser.add_argument('--data_type', type=str, required=True, choices=['expression', 'gesture'],
                        help='Type of data to collect (expression or gesture)')
    
    parser.add_argument('--label', type=str, required=True,
                        help='Label for the collected data (e.g., happy, thumbs_up)')
    
    parser.add_argument('--output_dir', type=str, default='data',
                        help='Directory to save the collected data')
    
    parser.add_argument('--num_samples', type=int, default=100,
                        help='Number of samples to collect')
    
    parser.add_argument('--camera', type=int, default=0,
                        help='Camera device index')
    
    parser.add_argument('--width', type=int, default=640,
                        help='Camera frame width')
    
    parser.add_argument('--height', type=int, default=480,
                        help='Camera frame height')
    
    return parser.parse_args()

def collect_expression_data(args):
    """
    Collect facial expression data.
    
    Args:
        args: Command line arguments.
    """
    # Create output directory
    output_dir = os.path.join(args.output_dir, 'expressions', args.label)
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize face detector
    face_detector = FaceDetector()
    
    # Initialize webcam
    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    
    # Initialize variables
    count = 0
    delay = 30  # Frames to wait between captures
    frame_count = 0
    
    print(f"Collecting {args.num_samples} samples of '{args.label}' expression...")
    print("Press 'q' to quit, 'c' to capture manually, or wait for automatic capture")
    
    while count < args.num_samples:
        # Read frame from webcam
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Failed to capture frame")
            break
        
        # Create a copy of the frame for processing
        frame_copy = frame.copy()
        
        # Detect faces
        frame_with_faces, face_rects = face_detector.detect_faces(frame_copy)
        
        # Extract face ROI
        face_roi = None
        if face_rects and len(face_rects) > 0:
            face_roi = face_detector.extract_face_roi(frame, face_rects)
        
        # Display the frame
        cv2.putText(frame_with_faces, f"Samples: {count}/{args.num_samples}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        if frame_count >= delay:
            cv2.putText(frame_with_faces, "CAPTURING...", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        cv2.imshow('Expression Data Collection', frame_with_faces)
        
        # Automatic capture after delay
        if frame_count >= delay and face_roi is not None:
            # Save the face ROI
            filename = os.path.join(output_dir, f"{args.label}_{count:04d}.npy")
            np.save(filename, face_roi)
            
            # Save a preview image
            preview_filename = os.path.join(output_dir, f"{args.label}_{count:04d}.jpg")
            cv2.imwrite(preview_filename, cv2.resize(face_roi * 255, (48, 48)))
            
            count += 1
            frame_count = 0
            print(f"Captured sample {count}/{args.num_samples}")
        else:
            frame_count += 1
        
        # Check for key press
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c') and face_roi is not None:
            # Manual capture
            filename = os.path.join(output_dir, f"{args.label}_{count:04d}.npy")
            np.save(filename, face_roi)
            
            # Save a preview image
            preview_filename = os.path.join(output_dir, f"{args.label}_{count:04d}.jpg")
            cv2.imwrite(preview_filename, cv2.resize(face_roi * 255, (48, 48)))
            
            count += 1
            frame_count = 0
            print(f"Manually captured sample {count}/{args.num_samples}")
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    
    print(f"Collected {count} samples of '{args.label}' expression")

def collect_gesture_data(args):
    """
    Collect hand gesture data.
    
    Args:
        args: Command line arguments.
    """
    # Create output directory
    output_dir = os.path.join(args.output_dir, 'gestures', args.label)
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize hand detector
    hand_detector = HandDetector()
    
    # Initialize webcam
    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    
    # Initialize variables
    count = 0
    delay = 30  # Frames to wait between captures
    frame_count = 0
    
    print(f"Collecting {args.num_samples} samples of '{args.label}' gesture...")
    print("Press 'q' to quit, 'c' to capture manually, 'r' to reset background, or wait for automatic capture")
    
    # Wait a moment to initialize background subtractor
    time.sleep(2)
    
    while count < args.num_samples:
        # Read frame from webcam
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Failed to capture frame")
            break
        
        # Create a copy of the frame for processing
        frame_copy = frame.copy()
        
        # Detect hands
        frame_with_hands, hand_rects = hand_detector.detect_hands(frame_copy)
        
        # Extract hand features
        hand_features = None
        if hand_rects and len(hand_rects) > 0:
            hand_features = hand_detector.get_hand_gesture_features(hand_rects)
        
        # Display the frame
        cv2.putText(frame_with_hands, f"Samples: {count}/{args.num_samples}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        if frame_count >= delay:
            cv2.putText(frame_with_hands, "CAPTURING...", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        cv2.imshow('Gesture Data Collection', frame_with_hands)
        
        # Automatic capture after delay
        if frame_count >= delay and hand_features and len(hand_features) > 0:
            # Save the hand features
            filename = os.path.join(output_dir, f"{args.label}_{count:04d}.npy")
            np.save(filename, hand_features[0])  # Save the first detected hand
            
            count += 1
            frame_count = 0
            print(f"Captured sample {count}/{args.num_samples}")
        else:
            frame_count += 1
        
        # Check for key press
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c') and hand_features and len(hand_features) > 0:
            # Manual capture
            filename = os.path.join(output_dir, f"{args.label}_{count:04d}.npy")
            np.save(filename, hand_features[0])  # Save the first detected hand
            
            count += 1
            frame_count = 0
            print(f"Manually captured sample {count}/{args.num_samples}")
        elif key == ord('r'):
            # Reset background subtractor
            hand_detector = HandDetector()
            print("Background reset for hand detection")
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    
    print(f"Collected {count} samples of '{args.label}' gesture")

def main():
    """
    Main function for data collection.
    """
    # Parse arguments
    args = parse_args()
    
    # Collect data based on type
    if args.data_type == 'expression':
        collect_expression_data(args)
    elif args.data_type == 'gesture':
        collect_gesture_data(args)

if __name__ == '__main__':
    main()
