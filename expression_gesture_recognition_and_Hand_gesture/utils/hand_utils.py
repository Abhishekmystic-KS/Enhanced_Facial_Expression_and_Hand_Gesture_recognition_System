import cv2
import numpy as np

class HandDetector:
    """
    A class for detecting hands using OpenCV's background subtraction and contour detection.
    """
    def __init__(self, history=500, threshold=400, detect_shadows=True):
        """
        Initialize the hand detector.

        Args:
            history: Length of history for background subtractor.
            threshold: Threshold for background subtractor.
            detect_shadows: Whether to detect shadows.
        """
        # Initialize background subtractor
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=history,
            varThreshold=threshold,
            detectShadows=detect_shadows
        )

        # Initialize variables for hand tracking
        self.hand_rects = []
        self.prev_hand_rects = []
        self.frame_count = 0

    def detect_hands(self, image):
        """
        Detect hands in an image using motion detection and contour analysis.

        Args:
            image: The input image (BGR format).

        Returns:
            processed_image: Image with hand rectangles drawn.
            hand_rects: Detected hand rectangles (x, y, w, h).
        """
        # Create a copy of the image for drawing
        processed_image = image.copy()

        # Apply background subtraction
        fg_mask = self.bg_subtractor.apply(image)

        # Apply thresholding to get binary image
        _, thresh = cv2.threshold(fg_mask, 127, 255, cv2.THRESH_BINARY)

        # Apply morphological operations to remove noise
        kernel = np.ones((5, 5), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Reset hand rectangles
        self.prev_hand_rects = self.hand_rects.copy()
        self.hand_rects = []

        # Process contours to find potential hands
        for contour in contours:
            # Filter small contours
            if cv2.contourArea(contour) < 1000:  # Adjust this threshold as needed
                continue

            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)

            # Filter by aspect ratio (hands are usually taller than wide)
            aspect_ratio = h / float(w)
            if aspect_ratio < 0.5 or aspect_ratio > 3.0:  # Adjust these thresholds as needed
                continue

            # Draw rectangle
            cv2.rectangle(processed_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Add to hand rectangles
            self.hand_rects.append((x, y, w, h))

        # If no hands detected, use previous detections for a few frames
        if len(self.hand_rects) == 0 and self.frame_count < 10:
            self.hand_rects = self.prev_hand_rects
            self.frame_count += 1
        else:
            self.frame_count = 0

        return processed_image, self.hand_rects

    def extract_hand_roi(self, image, hand_rects, target_size=(128, 128)):
        """
        Extract the hand region of interest (ROI) from the image.

        Args:
            image: The input image.
            hand_rects: Detected hand rectangles.
            target_size: Size to resize the extracted hand to.

        Returns:
            hand_rois: List of extracted and preprocessed hand regions.
        """
        if hand_rects is None or len(hand_rects) == 0:
            return None

        hand_rois = []

        for (x, y, w, h) in hand_rects:
            # Extract hand ROI
            hand_roi = image[y:y+h, x:x+w]

            # Resize to target size
            hand_roi = cv2.resize(hand_roi, target_size)

            # Convert to HSV for better color-based features
            hand_roi_hsv = cv2.cvtColor(hand_roi, cv2.COLOR_BGR2HSV)

            # Normalize pixel values
            hand_roi_normalized = hand_roi_hsv / 255.0

            hand_rois.append(hand_roi_normalized)

        return hand_rois

    def get_hand_gesture_features(self, hand_rects):
        """
        Extract features from hand regions for gesture recognition.

        Args:
            hand_rects: Detected hand rectangles.

        Returns:
            features: Extracted features for gesture recognition.
        """
        if hand_rects is None or len(hand_rects) == 0:
            return None

        features = []

        for (x, y, w, h) in hand_rects:
            # Simple geometric features
            aspect_ratio = h / float(w)
            area = w * h
            perimeter = 2 * (w + h)

            # Create a feature vector
            # We'll use a simplified feature vector with basic geometric properties
            # In a real application, you would extract more sophisticated features
            hand_features = [
                x, y, w, h,
                aspect_ratio,
                area,
                perimeter,
                w / float(h),  # inverse aspect ratio
                np.sqrt(area),  # square root of area
                # Add more features as needed to reach 63 dimensions
                # Padding with zeros for now to match the expected input shape
            ]

            # Pad to match the expected input shape (63 features)
            hand_features.extend([0] * (63 - len(hand_features)))

            features.append(hand_features)

        return features
