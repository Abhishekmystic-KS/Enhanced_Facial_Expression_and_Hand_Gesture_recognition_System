import cv2
import numpy as np
import math

class ImprovedHandDetector:
    """
    A class for improved hand detection and finger counting using OpenCV.
    """
    def __init__(self, threshold=60, blur_value=41):
        """
        Initialize the improved hand detector.
        
        Args:
            threshold: Threshold value for binary thresholding.
            blur_value: Blur kernel size for preprocessing.
        """
        self.threshold = threshold
        self.blur_value = blur_value
        self.background = None
        self.hand_rect_two_corner = [(0, 0), (0, 0)]
        self.hand_rect_one_corner = [(0, 0), (0, 0)]
        self.finger_count_history = []
        self.gesture_history = []
        
    def reset_background(self):
        """Reset the background model."""
        self.background = None
        
    def extract_hand_region(self, frame, x, y, w, h, margin=20):
        """
        Extract the hand region from the frame.
        
        Args:
            frame: Input frame.
            x, y, w, h: Bounding box coordinates.
            margin: Margin to add around the bounding box.
            
        Returns:
            hand_region: Extracted hand region.
            (x, y): Top-left corner coordinates.
        """
        # Get frame dimensions
        frame_h, frame_w = frame.shape[:2]
        
        # Add margin
        x1 = max(0, x - margin)
        y1 = max(0, y - margin)
        x2 = min(frame_w, x + w + margin)
        y2 = min(frame_h, y + h + margin)
        
        # Extract region
        hand_region = frame[y1:y2, x1:x2]
        
        return hand_region, (x1, y1)
        
    def detect_hand(self, frame, roi_x=None, roi_y=None, roi_w=None, roi_h=None):
        """
        Detect hand in the frame using background subtraction and contour analysis.
        
        Args:
            frame: Input frame.
            roi_x, roi_y, roi_w, roi_h: Region of interest coordinates (optional).
            
        Returns:
            processed_frame: Frame with hand detection visualization.
            hand_data: Dictionary with hand detection data.
        """
        # Create a copy of the frame
        processed_frame = frame.copy()
        
        # Define region of interest (ROI)
        if roi_x is not None and roi_y is not None and roi_w is not None and roi_h is not None:
            roi = frame[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
        else:
            h, w = frame.shape[:2]
            roi_x, roi_y = w // 4, h // 4
            roi_w, roi_h = w // 2, h // 2
            roi = frame[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
        
        # Draw ROI rectangle
        cv2.rectangle(processed_frame, (roi_x, roi_y), (roi_x+roi_w, roi_y+roi_h), (0, 255, 0), 2)
        
        # Convert ROI to grayscale
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blur = cv2.GaussianBlur(gray, (self.blur_value, self.blur_value), 0)
        
        # If background is not set, set it
        if self.background is None:
            self.background = blur.copy().astype("float")
            return processed_frame, {"hand_found": False, "finger_count": 0, "gesture": None}
        
        # Calculate absolute difference between background and current frame
        diff = cv2.absdiff(self.background.astype("uint8"), blur)
        
        # Apply threshold to get binary image
        _, thresh = cv2.threshold(diff, self.threshold, 255, cv2.THRESH_BINARY)
        
        # Apply morphological operations to clean up the image
        kernel = np.ones((3, 3), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        # Find contours
        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Initialize hand data
        hand_data = {"hand_found": False, "finger_count": 0, "gesture": None, "landmarks": []}
        
        if contours:
            # Get the largest contour (assumed to be the hand)
            max_contour = max(contours, key=cv2.contourArea)
            
            # Check if contour area is large enough
            if cv2.contourArea(max_contour) > 1000:  # Minimum area threshold
                hand_data["hand_found"] = True
                
                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(max_contour)
                
                # Draw bounding rectangle on the ROI
                cv2.rectangle(roi, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                # Extract hand region for further processing
                hand_region, (hand_x, hand_y) = self.extract_hand_region(roi, x, y, w, h)
                
                # Calculate global coordinates
                global_x = roi_x + hand_x
                global_y = roi_y + hand_y
                
                # Count fingers and detect gesture
                finger_count, landmarks = self.count_fingers(thresh, max_contour)
                hand_data["finger_count"] = finger_count
                
                # Convert landmarks to global coordinates
                global_landmarks = []
                for lm in landmarks:
                    global_landmarks.append((lm[0] + roi_x, lm[1] + roi_y))
                
                hand_data["landmarks"] = global_landmarks
                
                # Determine gesture based on finger count
                gesture = self.determine_gesture(finger_count, max_contour)
                hand_data["gesture"] = gesture
                
                # Draw finger count and gesture on the frame
                cv2.putText(processed_frame, f"Fingers: {finger_count}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(processed_frame, f"Gesture: {gesture}", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Draw landmarks on the frame
                for i, landmark in enumerate(global_landmarks):
                    cv2.circle(processed_frame, landmark, 5, (0, 0, 255), -1)
                    cv2.putText(processed_frame, str(i), landmark, 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Draw contour on the ROI
                cv2.drawContours(roi, [max_contour], 0, (0, 255, 0), 2)
        
        # Update background model (slow adaptation)
        cv2.accumulateWeighted(blur, self.background, 0.1)
        
        # Add binary threshold to the processed frame
        h, w = processed_frame.shape[:2]
        resized_thresh = cv2.resize(thresh, (w//4, h//4))
        processed_frame[0:h//4, 0:w//4] = cv2.cvtColor(resized_thresh, cv2.COLOR_GRAY2BGR)
        
        return processed_frame, hand_data
    
    def count_fingers(self, thresh, contour):
        """
        Count the number of fingers in the hand contour.
        
        Args:
            thresh: Thresholded image.
            contour: Hand contour.
            
        Returns:
            finger_count: Number of fingers detected.
            landmarks: List of finger tip coordinates.
        """
        # Find convex hull and convexity defects
        hull = cv2.convexHull(contour, returnPoints=False)
        
        # Check if hull has enough points
        if len(hull) < 3:
            return 0, []
        
        try:
            defects = cv2.convexityDefects(contour, hull)
        except:
            return 0, []
        
        if defects is None:
            return 0, []
        
        # Initialize finger count
        finger_count = 0
        
        # Initialize landmarks list
        landmarks = []
        
        # Process convexity defects
        for i in range(defects.shape[0]):
            s, e, f, d = defects[i, 0]
            start = tuple(contour[s][0])
            end = tuple(contour[e][0])
            far = tuple(contour[f][0])
            
            # Calculate triangle sides
            a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
            b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
            c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
            
            # Calculate angle
            angle = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c)) * 180 / math.pi
            
            # If angle is less than 90 degrees, it's likely a finger
            if angle <= 90:
                finger_count += 1
                landmarks.append(far)
        
        # Add 1 for the last finger (usually thumb)
        finger_count += 1
        
        # Add fingertips to landmarks
        hull_points = cv2.convexHull(contour, returnPoints=True)
        for point in hull_points:
            landmarks.append(tuple(point[0]))
        
        # Limit finger count to 5
        finger_count = min(finger_count, 5)
        
        return finger_count, landmarks
    
    def determine_gesture(self, finger_count, contour):
        """
        Determine the hand gesture based on finger count and contour shape.
        
        Args:
            finger_count: Number of fingers detected.
            contour: Hand contour.
            
        Returns:
            gesture: Detected gesture.
        """
        # Calculate contour features
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        
        # Calculate compactness (circularity)
        compactness = 4 * math.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
        
        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = float(w) / h if h > 0 else 0
        
        # Update gesture history
        self.finger_count_history.append(finger_count)
        if len(self.finger_count_history) > 10:
            self.finger_count_history.pop(0)
        
        # Get most common finger count in history
        if self.finger_count_history:
            from collections import Counter
            most_common_count = Counter(self.finger_count_history).most_common(1)[0][0]
        else:
            most_common_count = finger_count
        
        # Determine gesture based on finger count and shape
        if most_common_count == 0:
            gesture = "Fist"
        elif most_common_count == 1:
            if aspect_ratio < 0.5:
                gesture = "Pointing"
            else:
                gesture = "Thumbs Up" if y > 100 else "Thumbs Down"
        elif most_common_count == 2:
            gesture = "Peace"
        elif most_common_count == 3:
            gesture = "OK"
        elif most_common_count == 4:
            gesture = "Four"
        elif most_common_count == 5:
            gesture = "Open Palm"
        else:
            gesture = "Unknown"
        
        # Update gesture history
        self.gesture_history.append(gesture)
        if len(self.gesture_history) > 10:
            self.gesture_history.pop(0)
        
        # Get most common gesture in history
        if self.gesture_history:
            from collections import Counter
            most_common_gesture = Counter(self.gesture_history).most_common(1)[0][0]
        else:
            most_common_gesture = gesture
        
        return most_common_gesture
    
    def get_hand_features(self, hand_data):
        """
        Extract features from hand data for gesture recognition.
        
        Args:
            hand_data: Dictionary with hand detection data.
            
        Returns:
            features: Extracted features for gesture recognition.
        """
        if not hand_data["hand_found"]:
            return None
        
        # Basic features
        features = [
            hand_data["finger_count"],
            len(hand_data["landmarks"]),
        ]
        
        # Add landmark coordinates (normalized)
        if hand_data["landmarks"]:
            # Calculate centroid
            centroid_x = sum(lm[0] for lm in hand_data["landmarks"]) / len(hand_data["landmarks"])
            centroid_y = sum(lm[1] for lm in hand_data["landmarks"]) / len(hand_data["landmarks"])
            
            # Add normalized coordinates relative to centroid
            for lm in hand_data["landmarks"][:10]:  # Use up to 10 landmarks
                features.extend([
                    (lm[0] - centroid_x) / 100,  # Normalize
                    (lm[1] - centroid_y) / 100   # Normalize
                ])
            
            # Pad if less than 10 landmarks
            padding_needed = 10 - len(hand_data["landmarks"])
            if padding_needed > 0:
                features.extend([0] * (padding_needed * 2))
        else:
            # No landmarks, add zeros
            features.extend([0] * 20)  # 10 landmarks * 2 coordinates
        
        # Ensure we have 22 features (2 basic + 20 landmark coordinates)
        assert len(features) == 22, f"Expected 22 features, got {len(features)}"
        
        # Pad to match the expected input shape (63 features)
        features.extend([0] * (63 - len(features)))
        
        return features
