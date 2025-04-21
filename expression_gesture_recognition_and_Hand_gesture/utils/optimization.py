import cv2
import numpy as np
import time
import threading
from queue import Queue

class FrameProcessor:
    """
    A class for optimized frame processing using threading.
    """
    def __init__(self, max_queue_size=10):
        """
        Initialize the frame processor.
        
        Args:
            max_queue_size: Maximum size of the frame queue.
        """
        self.input_queue = Queue(maxsize=max_queue_size)
        self.output_queue = Queue(maxsize=max_queue_size)
        self.processing_thread = None
        self.running = False
        self.processing_time = 0
        self.fps = 0
        self.frame_count = 0
        self.start_time = time.time()
    
    def start(self, processing_function):
        """
        Start the frame processing thread.
        
        Args:
            processing_function: Function to process frames.
        """
        self.running = True
        self.processing_function = processing_function
        self.processing_thread = threading.Thread(target=self._process_frames)
        self.processing_thread.daemon = True
        self.processing_thread.start()
    
    def stop(self):
        """
        Stop the frame processing thread.
        """
        self.running = False
        if self.processing_thread is not None:
            self.processing_thread.join()
    
    def _process_frames(self):
        """
        Process frames from the input queue and put results in the output queue.
        """
        while self.running:
            if not self.input_queue.empty():
                # Get frame from input queue
                frame = self.input_queue.get()
                
                # Process frame
                start_time = time.time()
                result = self.processing_function(frame)
                self.processing_time = time.time() - start_time
                
                # Put result in output queue
                if not self.output_queue.full():
                    self.output_queue.put(result)
                
                # Update FPS
                self.frame_count += 1
                elapsed_time = time.time() - self.start_time
                if elapsed_time > 1.0:
                    self.fps = self.frame_count / elapsed_time
                    self.frame_count = 0
                    self.start_time = time.time()
            else:
                # Sleep to avoid high CPU usage
                time.sleep(0.001)
    
    def add_frame(self, frame):
        """
        Add a frame to the input queue.
        
        Args:
            frame: Frame to add.
            
        Returns:
            success: Whether the frame was added successfully.
        """
        if not self.input_queue.full():
            self.input_queue.put(frame)
            return True
        return False
    
    def get_result(self):
        """
        Get a result from the output queue.
        
        Returns:
            result: The processed result, or None if no result is available.
        """
        if not self.output_queue.empty():
            return self.output_queue.get()
        return None
    
    def get_stats(self):
        """
        Get processing statistics.
        
        Returns:
            stats: Dictionary with processing statistics.
        """
        return {
            'fps': self.fps,
            'processing_time': self.processing_time,
            'input_queue_size': self.input_queue.qsize(),
            'output_queue_size': self.output_queue.qsize()
        }

def resize_frame(frame, scale_percent=50):
    """
    Resize a frame for faster processing.
    
    Args:
        frame: Input frame.
        scale_percent: Scale percentage.
        
    Returns:
        resized_frame: Resized frame.
    """
    width = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized_frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
    return resized_frame

def optimize_detection_parameters(face_detector, hand_detector, frame):
    """
    Optimize detection parameters based on the frame.
    
    Args:
        face_detector: Face detector instance.
        hand_detector: Hand detector instance.
        frame: Input frame.
        
    Returns:
        optimized_face_detector: Optimized face detector.
        optimized_hand_detector: Optimized hand detector.
    """
    # Get frame dimensions
    h, w = frame.shape[:2]
    
    # Optimize face detector parameters based on frame size
    if w > 1280:
        # For high-resolution frames, use more aggressive scaling
        if hasattr(face_detector, 'face_cascade'):
            face_detector.face_cascade.setScaleFactor(1.2)
    elif w < 640:
        # For low-resolution frames, use less aggressive scaling
        if hasattr(face_detector, 'face_cascade'):
            face_detector.face_cascade.setScaleFactor(1.05)
    
    # Optimize hand detector parameters based on frame size
    if w > 1280:
        # For high-resolution frames, increase contour area threshold
        hand_detector.min_contour_area = 2000
    elif w < 640:
        # For low-resolution frames, decrease contour area threshold
        hand_detector.min_contour_area = 500
    
    return face_detector, hand_detector

class MotionDetector:
    """
    A class for detecting motion to optimize processing.
    """
    def __init__(self, threshold=25, min_area=500):
        """
        Initialize the motion detector.
        
        Args:
            threshold: Threshold for motion detection.
            min_area: Minimum contour area to consider as motion.
        """
        self.threshold = threshold
        self.min_area = min_area
        self.prev_frame = None
        self.motion_detected = False
        self.motion_regions = []
    
    def detect_motion(self, frame):
        """
        Detect motion in a frame.
        
        Args:
            frame: Input frame.
            
        Returns:
            motion_detected: Whether motion was detected.
            motion_regions: Regions with motion.
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        
        # Initialize previous frame if not set
        if self.prev_frame is None:
            self.prev_frame = gray
            return False, []
        
        # Compute absolute difference between current and previous frame
        frame_delta = cv2.absdiff(self.prev_frame, gray)
        
        # Apply threshold
        thresh = cv2.threshold(frame_delta, self.threshold, 255, cv2.THRESH_BINARY)[1]
        
        # Dilate the thresholded image to fill in holes
        thresh = cv2.dilate(thresh, None, iterations=2)
        
        # Find contours
        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Initialize motion regions
        self.motion_regions = []
        
        # Process contours
        for contour in contours:
            if cv2.contourArea(contour) < self.min_area:
                continue
            
            # Get bounding box
            (x, y, w, h) = cv2.boundingRect(contour)
            self.motion_regions.append((x, y, w, h))
        
        # Update previous frame
        self.prev_frame = gray
        
        # Update motion detected flag
        self.motion_detected = len(self.motion_regions) > 0
        
        return self.motion_detected, self.motion_regions

def optimize_processing_regions(frame, motion_regions, margin=20):
    """
    Optimize processing by focusing only on regions with motion.
    
    Args:
        frame: Input frame.
        motion_regions: Regions with motion.
        margin: Margin to add around motion regions.
        
    Returns:
        regions: List of frame regions to process.
    """
    if not motion_regions:
        # If no motion regions, process the entire frame
        return [frame]
    
    # Get frame dimensions
    h, w = frame.shape[:2]
    
    # Initialize regions
    regions = []
    
    # Process each motion region
    for (x, y, w_region, h_region) in motion_regions:
        # Add margin
        x_min = max(0, x - margin)
        y_min = max(0, y - margin)
        x_max = min(w, x + w_region + margin)
        y_max = min(h, y + h_region + margin)
        
        # Extract region
        region = frame[y_min:y_max, x_min:x_max]
        regions.append((region, (x_min, y_min)))
    
    return regions
