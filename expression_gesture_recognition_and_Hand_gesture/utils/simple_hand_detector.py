import cv2
import numpy as np
import time

class SimpleHandDetector:
    """
    A simple hand detector that creates a green bounding box with confidence display,
    similar to the reference image.
    """
    def __init__(self, threshold=20, blur_value=7, min_area=5000, max_area=50000):
        """
        Initialize the simple hand detector.
        
        Args:
            threshold: Threshold value for binary thresholding.
            blur_value: Blur kernel size for preprocessing.
            min_area: Minimum contour area to be considered a hand.
            max_area: Maximum contour area to be considered a hand.
        """
        self.threshold = threshold
        self.blur_value = blur_value
        self.min_area = min_area
        self.max_area = max_area
        self.background = None
        self.prev_time = 0
        self.current_time = 0
        
    def reset_background(self):
        """Reset the background model."""
        self.background = None
        
    def detect_hand(self, frame):
        """
        Detect hand in the frame and draw a green bounding box with confidence.
        
        Args:
            frame: Input frame.
            
        Returns:
            processed_frame: Frame with hand detection visualization.
            hand_data: Dictionary with hand detection data.
        """
        # Create a copy of the frame
        processed_frame = frame.copy()
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blur = cv2.GaussianBlur(gray, (self.blur_value, self.blur_value), 0)
        
        # Initialize background model if not already done
        if self.background is None:
            self.background = blur.copy().astype("float")
            return processed_frame, {"hand_found": False}
        
        # Calculate absolute difference between background and current frame
        diff = cv2.absdiff(self.background.astype("uint8"), blur)
        
        # Apply threshold to get binary image
        _, thresh = cv2.threshold(diff, self.threshold, 255, cv2.THRESH_BINARY)
        
        # Apply morphological operations to clean up the image
        kernel = np.ones((5, 5), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        # Find contours
        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Initialize hand data
        hand_data = {"hand_found": False, "gesture": "unknown", "confidence": 0.0}
        
        if contours:
            # Get the largest contour (assumed to be the hand)
            max_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(max_contour)
            
            # Check if contour area is within the expected range for a hand
            if self.min_area < area < self.max_area:
                hand_data["hand_found"] = True
                
                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(max_contour)
                
                # Calculate confidence based on contour area
                # Map area from min_area-max_area to 0.7-1.0 confidence range
                confidence = 0.7 + 0.3 * min(1.0, (area - self.min_area) / (self.max_area - self.min_area))
                hand_data["confidence"] = confidence
                
                # Determine gesture based on contour shape
                gesture = self.determine_gesture(max_contour)
                hand_data["gesture"] = gesture
                
                # Draw green bounding box
                cv2.rectangle(processed_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # Draw confidence label
                label = f"{gesture}: {int(confidence * 100)}%"
                cv2.putText(processed_frame, label, (x, y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # Update background model (slow adaptation)
        cv2.accumulateWeighted(blur, self.background, 0.1)
        
        return processed_frame, hand_data
    
    def determine_gesture(self, contour):
        """
        Determine the hand gesture based on contour shape.
        
        Args:
            contour: Hand contour.
            
        Returns:
            gesture: Detected gesture.
        """
        # Calculate convex hull and convexity defects
        hull = cv2.convexHull(contour, returnPoints=False)
        defects = None
        try:
            defects = cv2.convexityDefects(contour, hull)
        except:
            return "unknown"
        
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
                
                # Avoid division by zero
                if b * c == 0:
                    continue
                
                # Calculate angle using law of cosines
                angle = np.arccos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c))
                
                # If angle is less than 90 degrees, consider it a finger
                if angle <= np.pi / 2:
                    finger_count += 1
        
        # Adjust finger count (convexity defects give spaces between fingers)
        finger_count = min(finger_count + 1, 5)
        
        # Determine gesture based on finger count
        if finger_count == 5:
            return "hello"  # Open palm
        elif finger_count == 2:
            return "peace"  # Peace sign
        elif finger_count == 1:
            # Check if it's a thumbs up or pointing
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
