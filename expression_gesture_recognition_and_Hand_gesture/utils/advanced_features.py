import cv2
import numpy as np
from skimage.feature import hog, local_binary_pattern
from skimage.transform import integral_image
from skimage.feature import haar_like_feature_coord, haar_like_feature

def extract_hog_features(image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2)):
    """
    Extract Histogram of Oriented Gradients (HOG) features from an image.
    
    Args:
        image: Input image.
        orientations: Number of orientation bins.
        pixels_per_cell: Size (in pixels) of a cell.
        cells_per_block: Number of cells in each block.
        
    Returns:
        features: Extracted HOG features.
    """
    # Ensure image is normalized
    if image.max() > 1.0:
        image = image / 255.0
    
    # Extract HOG features
    features = hog(
        image,
        orientations=orientations,
        pixels_per_cell=pixels_per_cell,
        cells_per_block=cells_per_block,
        visualize=False,
        feature_vector=True
    )
    
    return features

def extract_lbp_features(image, radius=3, n_points=24, method='uniform'):
    """
    Extract Local Binary Pattern (LBP) features from an image.
    
    Args:
        image: Input image.
        radius: Radius of circle (spatial resolution of the operator).
        n_points: Number of circularly symmetric neighbor set points.
        method: Method to determine the pattern.
        
    Returns:
        features: Extracted LBP features.
    """
    # Ensure image is normalized
    if image.max() > 1.0:
        image = image / 255.0
    
    # Extract LBP features
    lbp = local_binary_pattern(image, n_points, radius, method)
    
    # Calculate histogram of LBP
    n_bins = int(lbp.max() + 1)
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
    
    return hist

def extract_haar_features(image, feature_type='type-2-x', n_features=100):
    """
    Extract Haar-like features from an image.
    
    Args:
        image: Input image.
        feature_type: Type of Haar-like feature.
        n_features: Number of features to extract.
        
    Returns:
        features: Extracted Haar-like features.
    """
    # Ensure image is normalized
    if image.max() > 1.0:
        image = image / 255.0
    
    # Convert to grayscale if needed
    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Calculate integral image
    int_img = integral_image(image)
    
    # Define feature types
    feature_types = [feature_type]
    
    # Extract Haar-like features
    feature_coord, _ = haar_like_feature_coord(
        width=image.shape[1],
        height=image.shape[0],
        feature_type=feature_types
    )
    
    # Select a subset of features
    if len(feature_coord) > n_features:
        indices = np.random.choice(len(feature_coord), n_features, replace=False)
        feature_coord = feature_coord[indices]
    
    # Calculate features
    features = haar_like_feature(int_img, 0, 0, image.shape[0], image.shape[1], feature_coord)
    
    return features

def extract_gabor_features(image, frequencies=(0.1, 0.25, 0.4), orientations=(0, 45, 90, 135)):
    """
    Extract Gabor filter features from an image.
    
    Args:
        image: Input image.
        frequencies: List of frequencies for Gabor filters.
        orientations: List of orientations (in degrees) for Gabor filters.
        
    Returns:
        features: Extracted Gabor filter features.
    """
    # Ensure image is normalized
    if image.max() > 1.0:
        image = image / 255.0
    
    # Convert to grayscale if needed
    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Initialize features
    features = []
    
    # Apply Gabor filters with different parameters
    for frequency in frequencies:
        for theta in orientations:
            # Convert theta to radians
            theta_rad = theta * np.pi / 180
            
            # Create Gabor kernel
            kernel = cv2.getGaborKernel(
                ksize=(21, 21),
                sigma=5.0,
                theta=theta_rad,
                lambd=1.0/frequency,
                gamma=0.5,
                psi=0
            )
            
            # Apply filter
            filtered = cv2.filter2D(image, cv2.CV_64F, kernel)
            
            # Extract statistics from filtered image
            mean = np.mean(filtered)
            std = np.std(filtered)
            
            features.extend([mean, std])
    
    return np.array(features)

def extract_advanced_face_features(face_image):
    """
    Extract advanced features from a face image.
    
    Args:
        face_image: Input face image.
        
    Returns:
        features: Combined advanced features.
    """
    # Extract different types of features
    hog_features = extract_hog_features(face_image)
    lbp_features = extract_lbp_features(face_image)
    gabor_features = extract_gabor_features(face_image)
    
    # Combine features
    combined_features = np.concatenate([
        hog_features,
        lbp_features,
        gabor_features
    ])
    
    return combined_features

def extract_advanced_hand_features(hand_rects, frame):
    """
    Extract advanced features from hand regions.
    
    Args:
        hand_rects: Detected hand rectangles.
        frame: Input frame.
        
    Returns:
        features: Advanced hand features.
    """
    if not hand_rects or len(hand_rects) == 0:
        return None
    
    features = []
    
    for (x, y, w, h) in hand_rects:
        # Extract hand ROI
        hand_roi = frame[y:y+h, x:x+w]
        
        # Resize to a standard size
        hand_roi = cv2.resize(hand_roi, (64, 64))
        
        # Convert to grayscale
        if len(hand_roi.shape) > 2:
            hand_roi_gray = cv2.cvtColor(hand_roi, cv2.COLOR_BGR2GRAY)
        else:
            hand_roi_gray = hand_roi
        
        # Normalize
        hand_roi_norm = hand_roi_gray / 255.0
        
        # Extract HOG features
        hog_features = extract_hog_features(hand_roi_norm)
        
        # Extract contour-based features
        contour_features = extract_contour_features(hand_roi_gray)
        
        # Combine features
        hand_features = np.concatenate([
            hog_features,
            contour_features,
            # Add basic geometric features
            [w / float(h), w * h, 2 * (w + h)]
        ])
        
        features.append(hand_features)
    
    return features

def extract_contour_features(image):
    """
    Extract contour-based features from an image.
    
    Args:
        image: Input image.
        
    Returns:
        features: Contour-based features.
    """
    # Ensure image is in the right format
    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)
    
    # Apply threshold
    _, thresh = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Initialize features
    features = []
    
    if contours:
        # Get the largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Calculate contour area
        area = cv2.contourArea(largest_contour)
        
        # Calculate contour perimeter
        perimeter = cv2.arcLength(largest_contour, True)
        
        # Calculate contour solidity
        hull = cv2.convexHull(largest_contour)
        hull_area = cv2.contourArea(hull)
        solidity = float(area) / hull_area if hull_area > 0 else 0
        
        # Calculate contour extent
        x, y, w, h = cv2.boundingRect(largest_contour)
        rect_area = w * h
        extent = float(area) / rect_area if rect_area > 0 else 0
        
        # Calculate contour aspect ratio
        aspect_ratio = float(w) / h if h > 0 else 0
        
        # Add features
        features.extend([
            area,
            perimeter,
            solidity,
            extent,
            aspect_ratio
        ])
    else:
        # If no contours found, add zeros
        features.extend([0, 0, 0, 0, 0])
    
    return np.array(features)
