import cv2
import numpy as np
import os
import argparse
import time
import math
import mediapipe as mp
from utils.face_utils import FaceDetector
from utils.hand_utils import HandDetector
from utils.visualization import create_prediction_overlay
from models.expression_model import ExpressionRecognitionModel
from models.gesture_model import GestureRecognitionModel

# Argument parsing function
def parse_args():
    """
    Parse command line arguments.
    Returns:
        args: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description='Facial Expression and Hand Gesture Recognition')
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
    return parser.parse_args()

# Gesture classification function
def classify_gesture(landmarks):
    """
    Classify hand gesture based on MediaPipe landmarks.
    Args:
        landmarks: List of hand landmarks from MediaPipe.
    Returns:
        gesture: Detected gesture.
        confidence: Confidence score for the gesture.
    """
    # Extract finger tip and base coordinates
    thumb_tip, thumb_base = landmarks[4], landmarks[2]
    index_finger_tip, index_finger_base = landmarks[8], landmarks[5]
    middle_finger_tip, middle_finger_base = landmarks[12], landmarks[9]
    ring_finger_tip, ring_finger_base = landmarks[16], landmarks[13]
    pinky_tip, pinky_base = landmarks[20], landmarks[17]
    wrist = landmarks[0]

    # Calculate finger states (extended or not)
    # Use a threshold to make detection more accurate
    threshold = 0.04  # Slightly reduced threshold for better detection
    thumb_threshold = 0.08  # Higher threshold for thumb to make it more strict

    # Check if fingers are extended by comparing tip and base positions
    index_extended = index_finger_tip.y < (index_finger_base.y - threshold)
    middle_extended = middle_finger_tip.y < (middle_finger_base.y - threshold)
    ring_extended = ring_finger_tip.y < (ring_finger_base.y - threshold)
    pinky_extended = pinky_tip.y < (pinky_base.y - threshold)
    thumb_extended = thumb_tip.y < (thumb_base.y - thumb_threshold) or thumb_tip.x < (thumb_base.x - thumb_threshold)

    # Calculate distance between thumb tip and index finger tip
    thumb_index_distance = ((thumb_tip.x - index_finger_tip.x) ** 2 +
                           (thumb_tip.y - index_finger_tip.y) ** 2) ** 0.5

    # A small distance indicates thumb and index finger are touching

    # Calculate the angle between index and middle fingers for peace sign detection
    index_vector_x = index_finger_tip.x - index_finger_base.x
    index_vector_y = index_finger_tip.y - index_finger_base.y
    middle_vector_x = middle_finger_tip.x - middle_finger_base.x
    middle_vector_y = middle_finger_tip.y - middle_finger_base.y

    # Calculate dot product and magnitudes
    dot_product = index_vector_x * middle_vector_x + index_vector_y * middle_vector_y
    index_magnitude = math.sqrt(index_vector_x**2 + index_vector_y**2)
    middle_magnitude = math.sqrt(middle_vector_x**2 + middle_vector_y**2)

    # Calculate angle between vectors (in radians)
    if index_magnitude > 0 and middle_magnitude > 0:
        angle = math.acos(max(-1.0, min(1.0, dot_product / (index_magnitude * middle_magnitude))))
    else:
        angle = 0

    # Calculate angle between index and middle fingers for peace sign detection
    # This helps with debugging but is not used in the actual gesture detection

    # Print finger states for debugging
    print(f"Finger states: I:{index_extended} M:{middle_extended} R:{ring_extended} P:{pinky_extended} T:{thumb_extended} Touch:{thumb_index_distance:.3f} Angle:{angle:.2f}")

    # Peace gesture (index and middle fingers extended, rest closed)
    # This is a high priority gesture, so we check it first
    # We're more lenient with the thumb position for peace sign
    if (index_extended and
        middle_extended and
        not ring_extended and
        not pinky_extended):
        # Check if the angle between index and middle fingers is appropriate for a peace sign
        # If the angle is wide enough, it's more likely to be a peace sign
        if angle > 0.2:  # Approximately 11 degrees or more
            return "Peace", 0.98
        else:
            # Still a peace sign but with lower confidence if fingers are close together
            return "Peace", 0.95

    # Alternative peace sign detection that allows thumb to be extended
    # This is common in many people's peace sign gesture
    elif (index_extended and
          middle_extended and
          not ring_extended and
          not pinky_extended and
          thumb_extended):
        # Check if the angle between index and middle fingers is appropriate
        if angle > 0.2:  # Approximately 11 degrees or more
            return "Peace", 0.97
        else:
            return "Peace", 0.94

    # I Love You gesture (pinky, index finger, and thumb extended, others folded)
    elif (index_extended and
          pinky_extended and
          thumb_extended and
          not middle_extended and
          not ring_extended):
        return "I Love You", 0.98

    # Super gesture (middle, ring, pinky extended, thumb and index touching)
    elif (middle_extended and
          ring_extended and
          pinky_extended and
          thumb_index_distance < 0.03):
        return "Super", 0.98

    # Three fingers gesture
    elif (index_extended and
          middle_extended and
          ring_extended and
          not pinky_extended and
          not thumb_extended):
        return "Three", 0.95

    # Thumbs Up
    elif (thumb_extended and
          thumb_tip.y < index_finger_tip.y and
          not index_extended and
          not middle_extended and
          not ring_extended and
          not pinky_extended):
        return "thumbs up", 0.95

    # Thumbs Down
    elif (thumb_tip.y > wrist.y and
          thumb_tip.y > index_finger_tip.y and
          not index_extended and
          not middle_extended):
        return "thumbs down", 0.95

    # Open Palm (hello)
    elif (index_extended and
          middle_extended and
          ring_extended and
          pinky_extended):
        return "hello", 0.90

    # Closed Fist
    elif (not index_extended and
          not middle_extended and
          not ring_extended and
          not pinky_extended and
          not thumb_extended):
        return "fist", 0.95

    # Default: Unknown gesture
    return "unknown", 0.70

# Main function
def main():
    """
    Main function for the application.
    """
    args = parse_args()

    # Initialize detectors and models
    face_detector = FaceDetector(detection_method=args.face_detection)
    hand_detector = HandDetector()

    # Set default paths for trained models if not provided
    if args.expression_model is None:
        expression_model_path = os.path.join('..', 'trained_models', 'expression_model.pkl')
        if os.path.exists(expression_model_path):
            args.expression_model = expression_model_path
            print(f"Using trained expression model: {expression_model_path}")

    if args.gesture_model is None:
        gesture_model_path = os.path.join('..', 'trained_models', 'gesture_model.pkl')
        if os.path.exists(gesture_model_path):
            args.gesture_model = gesture_model_path
            print(f"Using trained gesture model: {gesture_model_path}")

    # Initialize models with the specified paths
    expression_model = ExpressionRecognitionModel(model_path=args.expression_model)
    gesture_model = GestureRecognitionModel(model_path=args.gesture_model)

    # Update class labels to match your trained models
    # These should match the labels you used during training
    expression_model.class_labels = ['Happy', 'Sad', 'Surprise', 'Shock', 'Wink', 'Depressed']
    gesture_model.class_labels = ['Thumbs Up', 'Thumbs Down', 'Super', 'I Love You', 'Peace']

    # Print the available expressions and gestures
    print("Available expressions:", expression_model.class_labels)
    print("Available gestures:", gesture_model.class_labels)

    # Initialize MediaPipe Hands for multiple hands
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=4, min_detection_confidence=0.5, min_tracking_confidence=0.5)

    # Initialize webcam
    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    prev_time = time.time()

    print("Press 'q' to quit")
    print("Press 'r' to reset background for hand detection")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame")
            break

        # Create a copy of the frame for processing
        frame_copy = frame.copy()

        # Detect faces and predict expressions
        frame_with_faces, face_rects = face_detector.detect_faces(frame_copy)
        face_expression, face_confidence = None, None
        if face_rects and len(face_rects) > 0:
            face_roi = face_detector.extract_face_roi(frame, face_rects)
            if face_roi is not None:
                face_expression, face_confidence = expression_model.predict(face_roi)

        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame_with_faces, cv2.COLOR_BGR2RGB)

        # Process hands with MediaPipe
        results = hands.process(rgb_frame)

        # Initialize variables for hand gestures
        hand_gestures = []
        frame_with_hands = frame_with_faces.copy()

        # Calculate FPS
        current_time = time.time()
        fps = 1 / (current_time - prev_time) if 'prev_time' in locals() and (current_time - prev_time) > 0 else 0
        prev_time = current_time

        # Display FPS
        cv2.putText(frame_with_hands, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Check if MediaPipe detected any hands
        if results.multi_hand_landmarks:
            for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                # Classify the gesture using MediaPipe landmarks
                hand_gesture, hand_confidence = classify_gesture(hand_landmarks.landmark)

                # Get handedness (left or right hand)
                handedness = "Unknown"
                if results.multi_handedness and len(results.multi_handedness) > hand_idx:
                    handedness = results.multi_handedness[hand_idx].classification[0].label

                # Store the gesture information
                hand_gestures.append({
                    'gesture': hand_gesture,
                    'confidence': hand_confidence,
                    'landmarks': hand_landmarks,
                    'handedness': handedness
                })

                # Draw landmarks with different colors for each hand
                hand_colors = [(0, 255, 0), (0, 0, 255), (255, 0, 0), (255, 255, 0)]
                color_idx = hand_idx % len(hand_colors)
                mp_drawing.draw_landmarks(
                    frame_with_hands,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=hand_colors[color_idx], thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=hand_colors[color_idx], thickness=2)
                )

                # Draw bounding box around the hand
                h, w, _ = frame_with_hands.shape
                x_min = int(min(lm.x for lm in hand_landmarks.landmark) * w)
                y_min = int(min(lm.y for lm in hand_landmarks.landmark) * h)
                x_max = int(max(lm.x for lm in hand_landmarks.landmark) * w)
                y_max = int(max(lm.y for lm in hand_landmarks.landmark) * h)

                padding = 20
                x_min, y_min = max(0, x_min - padding), max(0, y_min - padding)
                x_max, y_max = min(w, x_max + padding), min(h, y_max + padding)

                cv2.rectangle(frame_with_hands, (x_min, y_min), (x_max, y_max), hand_colors[color_idx], 2)
                cv2.putText(
                    frame_with_hands,
                    f"{hand_gesture}: {int(hand_confidence * 100)}%",
                    (x_min, y_min - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    hand_colors[color_idx],
                    2
                )

        # If no hands are detected by MediaPipe, fall back to the original hand detector
        if not hand_gestures:
            frame_with_hands, hand_rects = hand_detector.detect_hands(frame_with_faces)
            if hand_rects and len(hand_rects) > 0:
                for hand_rect in hand_rects:
                    hand_features = hand_detector.get_hand_gesture_features([hand_rect])
                    if hand_features and len(hand_features) > 0:
                        hand_gesture, hand_confidence = gesture_model.predict(hand_features[0])

                        # Guess handedness based on position
                        x, _, w, _ = hand_rect
                        frame_width = frame_with_hands.shape[1]
                        handedness = "Left" if (x + w / 2) > frame_width / 2 else "Right"

                        hand_gestures.append({
                            'gesture': hand_gesture,
                            'confidence': hand_confidence,
                            'rect': hand_rect,
                            'handedness': handedness
                        })

        # Create prediction overlay with all detected hands
        result_frame = create_prediction_overlay(
            frame_with_hands,
            face_expression=face_expression,
            face_confidence=face_confidence,
            hand_gestures=hand_gestures
        )

        # Add instructions
        cv2.putText(result_frame, "Press 'q' to quit", (10, result_frame.shape[0] - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(result_frame, "Press 'r' to reset background", (10, result_frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Display the result
        cv2.imshow('Facial Expression and Hand Gesture Recognition', result_frame)

        # Check for key press
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            hand_detector = HandDetector()
            print("Background reset for hand detection")
            hands = mp_hands.Hands(static_image_mode=False, max_num_hands=4, min_detection_confidence=0.5, min_tracking_confidence=0.5)

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()