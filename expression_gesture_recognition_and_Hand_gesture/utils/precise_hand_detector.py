import cv2
import numpy as np
import time

class PreciseHandDetector:
    """
    A class for precise hand detection with bounding box and confidence display,
    similar to the reference image. This version uses OpenCV's built-in methods
    instead of MediaPipe to avoid compatibility issues.
    """
    def __init__(self, static_image_mode=False, max_num_hands=2,
                 min_detection_confidence=0.5, min_tracking_confidence=0.5):
        """
        Initialize the precise hand detector.

        Args:
            static_image_mode: Whether to treat the input images as a batch of static images.
            max_num_hands: Maximum number of hands to detect.
            min_detection_confidence: Minimum confidence value for hand detection.
            min_tracking_confidence: Minimum confidence value for hand tracking.
        """
        # Initialize variables for FPS calculation
        self.prev_time = 0
        self.current_time = 0

        # Initialize variables for gesture recognition
        self.gestures = {
            "hello": "Open palm facing camera",
            "peace": "Victory sign",
            "thumbs_up": "Thumbs up gesture",
            "fist": "Closed fist",
            "pointing": "Index finger pointing"
        }

        # Parameters for hand detection
        self.min_detection_confidence = min_detection_confidence
        self.max_num_hands = max_num_hands

        # Initialize background subtractor for motion detection
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=50)

        # Initialize results placeholder
        self.results = None

    def find_hands(self, img, draw=True, draw_box=True, label_box=True):
        """
        Find hands in an image using OpenCV methods.

        Args:
            img: Input image (BGR format).
            draw: Whether to draw hand contours.
            draw_box: Whether to draw bounding box around the hand.
            label_box: Whether to label the bounding box with confidence.

        Returns:
            img: Image with hand detection visualization.
            results: Hand detection results.
        """
        # Create a copy of the original image
        original_img = img.copy()

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Apply background subtraction to detect motion
        fg_mask = self.bg_subtractor.apply(blurred)

        # Apply thresholding to get binary image
        _, thresh = cv2.threshold(fg_mask, 127, 255, cv2.THRESH_BINARY)

        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Sort contours by area (largest first)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        # Store results
        self.results = {'contours': contours, 'thresh': thresh}

        # Process the largest contours (potential hands)
        processed_contours = 0
        for contour in contours:
            # Filter small contours
            if cv2.contourArea(contour) < 1000:
                continue

            if processed_contours >= self.max_num_hands:
                break

            # Calculate confidence based on contour area
            area = cv2.contourArea(contour)
            max_area = img.shape[0] * img.shape[1] * 0.25  # 25% of image
            confidence = min(area / max_area, 1.0) * self.min_detection_confidence

            if confidence < self.min_detection_confidence:
                continue

            processed_contours += 1

            if draw:
                # Draw contour
                cv2.drawContours(img, [contour], -1, (0, 255, 0), 2)

            if draw_box:
                # Get bounding box
                x, y, w, h = cv2.boundingRect(contour)

                # Add padding
                padding = 20
                x = max(0, x - padding)
                y = max(0, y - padding)
                w = min(img.shape[1] - x, w + 2*padding)
                h = min(img.shape[0] - y, h + 2*padding)

                # Draw bounding box
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

                if label_box:
                    # Determine gesture based on contour shape
                    gesture = self.recognize_gesture_from_contour(contour)

                    # Draw confidence label
                    label = f"{gesture}: {int(confidence * 100)}%"
                    cv2.putText(img, label, (x, y - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        return img, self.results

    def find_positions(self, img, hand_idx=0):
        """
        Find the positions of hand contour points.

        Args:
            img: Input image.
            hand_idx: Index of the hand to find positions for.

        Returns:
            position_list: List of contour point positions.
        """
        position_list = []

        if 'contours' in self.results and len(self.results['contours']) > hand_idx:
            contour = self.results['contours'][hand_idx]

            # Get contour points
            for i, point in enumerate(contour):
                x, y = point[0]
                position_list.append([i, x, y])

        return position_list

    def recognize_gesture_from_contour(self, contour):
        """
        Recognize hand gesture based on contour shape.

        Args:
            contour: Hand contour.

        Returns:
            gesture: Recognized gesture.
        """
        # Calculate contour features
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)

        # Calculate convex hull and convexity defects
        hull = cv2.convexHull(contour, returnPoints=False)
        defects = None
        try:
            defects = cv2.convexityDefects(contour, hull)
        except:
            pass

        # Count fingers based on convexity defects
        finger_count = 0
        if defects is not None:
            for i in range(defects.shape[0]):
                s, e, f, d = defects[i, 0]
                start = tuple(contour[s][0])
                end = tuple(contour[e][0])
                far = tuple(contour[f][0])

                # Calculate angle between fingers
                a = np.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
                b = np.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
                c = np.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
                angle = np.arccos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c))

                # If angle is less than 90 degrees, consider it a finger
                if angle <= np.pi / 2:
                    finger_count += 1

        # Adjust finger count (convexity defects give spaces between fingers)
        finger_count = min(finger_count + 1, 5)

        # Determine gesture based on finger count and shape
        if finger_count == 5:
            return "hello"  # Open palm
        elif finger_count == 2:
            return "peace"  # Peace sign
        elif finger_count == 1:
            # Check if it's a thumbs up or pointing
            # This is a simplified check based on contour orientation
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w) / h
            if aspect_ratio > 0.8:
                return "thumbs_up"
            else:
                return "pointing"
        elif finger_count == 0:
            return "fist"  # Closed fist

        return "unknown"

    def calculate_fps(self, img):
        """
        Calculate and display FPS on the image.

        Args:
            img: Input image.

        Returns:
            img: Image with FPS display.
            fps: Calculated FPS.
        """
        self.current_time = time.time()
        fps = 1 / (self.current_time - self.prev_time) if (self.current_time - self.prev_time) > 0 else 0
        self.prev_time = self.current_time

        cv2.putText(img, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        return img, fps
