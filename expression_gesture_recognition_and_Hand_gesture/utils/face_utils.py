import cv2
import numpy as np
import os

class FaceDetector:
    """
    A class for detecting faces using OpenCV's Haar Cascade or DNN face detector.
    """
    def __init__(self, detection_method='haar', min_confidence=0.5):
        """
        Initialize the face detector.

        Args:
            detection_method: Method to use for face detection ('haar' or 'dnn').
            min_confidence: Minimum confidence value for face detection (for DNN method).
        """
        self.detection_method = detection_method
        self.min_confidence = min_confidence

        if detection_method == 'haar':
            # Load the Haar Cascade face detector
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
        elif detection_method == 'dnn':
            # Load the DNN face detector
            prototxt_path = os.path.join('models', 'deploy.prototxt')
            model_path = os.path.join('models', 'res10_300x300_ssd_iter_140000.caffemodel')

            # Check if model files exist, if not use Haar Cascade as fallback
            if os.path.exists(prototxt_path) and os.path.exists(model_path):
                self.face_net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
            else:
                print("DNN model files not found. Using Haar Cascade as fallback.")
                self.detection_method = 'haar'
                cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                self.face_cascade = cv2.CascadeClassifier(cascade_path)
        else:
            raise ValueError("Invalid detection method. Use 'haar' or 'dnn'.")

    def detect_faces(self, image):
        """
        Detect faces in an image.

        Args:
            image: The input image (BGR format).

        Returns:
            processed_image: Image with face rectangles drawn.
            face_rects: Detected face rectangles (x, y, w, h) or (x1, y1, x2, y2).
        """
        # Create a copy of the image for drawing
        processed_image = image.copy()
        face_rects = []

        if self.detection_method == 'haar':
            # Convert to grayscale for Haar Cascade
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Detect faces
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )

            # Draw rectangles around faces
            for (x, y, w, h) in faces:
                cv2.rectangle(processed_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                face_rects.append((x, y, w, h))

        elif self.detection_method == 'dnn':
            # Get image dimensions
            (h, w) = image.shape[:2]

            # Create a blob from the image
            blob = cv2.dnn.blobFromImage(
                cv2.resize(image, (300, 300)), 1.0,
                (300, 300), (104.0, 177.0, 123.0)
            )

            # Pass the blob through the network
            self.face_net.setInput(blob)
            detections = self.face_net.forward()

            # Process detections
            for i in range(0, detections.shape[2]):
                confidence = detections[0, 0, i, 2]

                # Filter out weak detections
                if confidence > self.min_confidence:
                    # Compute the bounding box coordinates
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")

                    # Draw the bounding box
                    cv2.rectangle(processed_image, (startX, startY), (endX, endY), (0, 255, 0), 2)
                    face_rects.append((startX, startY, endX - startX, endY - startY))

        return processed_image, face_rects

    def extract_face_roi(self, image, face_rects, target_size=(48, 48)):
        """
        Extract the face region of interest (ROI) from the image.

        Args:
            image: The input image.
            face_rects: Detected face rectangles.
            target_size: Size to resize the extracted face to.

        Returns:
            face_roi: Extracted and preprocessed face region.
        """
        if face_rects is None or len(face_rects) == 0:
            return None

        # Use the first detected face
        (x, y, w, h) = face_rects[0]

        # Extract face ROI
        face_roi = image[y:y+h, x:x+w]

        # Resize to target size
        face_roi = cv2.resize(face_roi, target_size)

        # Convert to grayscale (for expression recognition)
        face_roi_gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)

        # Normalize pixel values
        face_roi_normalized = face_roi_gray / 255.0

        return face_roi_normalized
