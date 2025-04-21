import cv2
import os
import numpy as np
import argparse
import time
import mediapipe as mp
from utils.face_utils import FaceDetector
from utils.hand_utils import HandDetector

def parse_args():
    """
    Parse command line arguments.

    Returns:
        args: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description='Collect Training Data for Expression and Gesture Recognition')

    parser.add_argument('--data_type', type=str, required=True, choices=['expression', 'gesture'],
                        help='Type of data to collect (expression or gesture)')

    parser.add_argument('--output_dir', type=str, default='data',
                        help='Directory to save collected data')

    parser.add_argument('--camera', type=int, default=0,
                        help='Camera device index')

    parser.add_argument('--width', type=int, default=640,
                        help='Camera frame width')

    parser.add_argument('--height', type=int, default=480,
                        help='Camera frame height')

    parser.add_argument('--face_detection', type=str, default='haar', choices=['haar', 'dnn'],
                        help='Face detection method (haar or dnn)')

    parser.add_argument('--num_samples', type=int, default=50,
                        help='Number of samples to collect for each class')

    return parser.parse_args()

def collect_expression_data(args):
    """
    Collect facial expression data.

    Args:
        args: Command line arguments.
    """
    # Initialize face detector
    face_detector = FaceDetector(detection_method=args.face_detection)

    # Initialize webcam
    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    # Create output directory if it doesn't exist
    output_dir = os.path.join(args.output_dir, 'expression')
    os.makedirs(output_dir, exist_ok=True)

    # Define expression classes
    expression_classes = [
        'Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral', 
        'Contempt', 'Confused', 'Calm', 'Shocked', 'Wink', 'Depressed'
    ]

    # Collect data for each expression
    for expression in expression_classes:
        # Create directory for this expression
        expression_dir = os.path.join(output_dir, expression)
        os.makedirs(expression_dir, exist_ok=True)

        # Count existing samples
        existing_samples = len([f for f in os.listdir(expression_dir) if f.endswith('.jpg')])
        samples_to_collect = max(0, args.num_samples - existing_samples)

        if samples_to_collect == 0:
            print(f"Already collected {args.num_samples} samples for {expression}. Skipping.")
            continue

        print(f"Collecting {samples_to_collect} samples for expression: {expression}")
        print(f"Press 'c' to capture, 's' to skip to next expression, 'q' to quit")
        print(f"Make the {expression} expression and press 'c' to capture")

        sample_count = existing_samples
        countdown_active = False
        countdown_time = 0

        while sample_count < args.num_samples:
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

            # Display instructions
            cv2.putText(frame_with_faces, f"Expression: {expression}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame_with_faces, f"Samples: {sample_count}/{args.num_samples}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame_with_faces, "Press 'c' to capture", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Handle countdown
            if countdown_active:
                remaining = max(0, 3 - int(time.time() - countdown_time))
                if remaining > 0:
                    cv2.putText(frame_with_faces, f"Capturing in {remaining}...", (10, 120),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                else:
                    countdown_active = False
                    if face_roi is not None:
                        # Save the face ROI
                        filename = os.path.join(expression_dir, f"{expression}_{sample_count}.jpg")
                        cv2.imwrite(filename, face_roi)
                        print(f"Saved {filename}")
                        sample_count += 1
                    else:
                        print("No face detected. Try again.")

            # Display the frame
            cv2.imshow('Collect Expression Data', frame_with_faces)

            # Check for key press
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c') and not countdown_active:
                if face_roi is not None:
                    countdown_active = True
                    countdown_time = time.time()
                else:
                    print("No face detected. Try again.")
            elif key == ord('s'):
                break

        if key == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

def collect_gesture_data(args):
    """
    Collect hand gesture data.

    Args:
        args: Command line arguments.
    """
    # Initialize MediaPipe Hands
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.5)

    # Initialize webcam
    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    # Create output directory if it doesn't exist
    output_dir = os.path.join(args.output_dir, 'gesture')
    os.makedirs(output_dir, exist_ok=True)

    # Define gesture classes
    gesture_classes = [
        'thumbs up', 'thumbs down', 'super', 'I love you', 'hello', 'fist', 'peace', 'pointing'
    ]

    # Collect data for each gesture
    for gesture in gesture_classes:
        # Create directory for this gesture
        gesture_dir = os.path.join(output_dir, gesture.replace(' ', '_'))
        os.makedirs(gesture_dir, exist_ok=True)

        # Count existing samples
        existing_samples = len([f for f in os.listdir(gesture_dir) if f.endswith('.npy')])
        samples_to_collect = max(0, args.num_samples - existing_samples)

        if samples_to_collect == 0:
            print(f"Already collected {args.num_samples} samples for {gesture}. Skipping.")
            continue

        print(f"Collecting {samples_to_collect} samples for gesture: {gesture}")
        print(f"Press 'c' to capture, 's' to skip to next gesture, 'q' to quit")
        print(f"Make the {gesture} gesture and press 'c' to capture")

        sample_count = existing_samples
        countdown_active = False
        countdown_time = 0

        while sample_count < args.num_samples:
            # Read frame from webcam
            ret, frame = cap.read()

            if not ret:
                print("Error: Failed to capture frame")
                break

            # Convert the BGR image to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process the frame with MediaPipe Hands
            results = hands.process(rgb_frame)

            # Create a copy of the frame for visualization
            frame_with_hands = frame.copy()

            # Check if MediaPipe detected any hands
            hand_landmarks = None
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw landmarks
                    mp_drawing.draw_landmarks(
                        frame_with_hands,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                        mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)
                    )

            # Display instructions
            cv2.putText(frame_with_hands, f"Gesture: {gesture}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame_with_hands, f"Samples: {sample_count}/{args.num_samples}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame_with_hands, "Press 'c' to capture", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Handle countdown
            if countdown_active:
                remaining = max(0, 3 - int(time.time() - countdown_time))
                if remaining > 0:
                    cv2.putText(frame_with_hands, f"Capturing in {remaining}...", (10, 120),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                else:
                    countdown_active = False
                    if results.multi_hand_landmarks:
                        # Extract hand landmarks
                        landmarks_array = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten()
                        
                        # Save the landmarks
                        filename = os.path.join(gesture_dir, f"{gesture.replace(' ', '_')}_{sample_count}.npy")
                        np.save(filename, landmarks_array)
                        print(f"Saved {filename}")
                        sample_count += 1
                    else:
                        print("No hand detected. Try again.")

            # Display the frame
            cv2.imshow('Collect Gesture Data', frame_with_hands)

            # Check for key press
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c') and not countdown_active:
                if results.multi_hand_landmarks:
                    countdown_active = True
                    countdown_time = time.time()
                else:
                    print("No hand detected. Try again.")
            elif key == ord('s'):
                break

        if key == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

def main():
    """
    Main function for collecting training data.
    """
    # Parse arguments
    args = parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Collect data based on the specified type
    if args.data_type == 'expression':
        collect_expression_data(args)
    elif args.data_type == 'gesture':
        collect_gesture_data(args)

if __name__ == '__main__':
    main()
